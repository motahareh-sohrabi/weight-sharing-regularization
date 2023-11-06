import os
import time

import submitit
import torch
import torch.nn as nn
import wandb
from absl import app
from absl.flags import FLAGS
from ml_collections.config_flags import config_flags as MLC_FLAGS
from torch.utils.data import DataLoader

from src.dataset import get_dataset
from src.models import get_model
from src.utils.train_util import rho_scheduler, train_epoch
from src.utils.utils import (
    calculate_accuracy,
    compute_distinct_non_zero_weights,
    compute_sparsity,
    get_heatmap_fig,
    set_seed,
)

DEFAULT_EXCLUDE = "kepler5,cn-b[001-005],cn-e[002-003],cn-g[005-012,017-026],cn-j001"
# ,cn-a[001-011],cn-c[001-040]
# Initialize "FLAGS.config" object with a placeholder config
MLC_FLAGS.DEFINE_config_file("config", default="src/configs/cifar10.py")
DEVICE = "cuda"  # if torch.cuda.is_available() else "cpu"


class Trainer:
    def __init__(self, config):
        self.config = config

    def __call__(self):
        self._make_reproducible()

        self.train_dataset, self.val_dataset = self._create_datasets()

        self.train_loader, self.val_loader = self._create_dataloaders()

        self.model = self._create_model()

        self.optimizer, self.scheduler = self._create_optimizer()

        self.loss_fn = self._create_loss_fn()

        self.device = DEVICE

        if (
            hasattr(self.config, "run_checkpoint_dir")
            and self.config.run_checkpoint_dir != "None"
        ):
            custom_run_id = self.config.run_checkpoint_dir
            # self.run_checkpoint_dir = self.config.run_checkpoint_dir
            # print(f"Overriding run_checkpoint_dir to {self.run_checkpoint_dir}")
            # print(os.path.join(self.run_checkpoint_dir, "checkpoint.pth"))
        else:
            custom_run_id = None

        self.wandb_run, self.run_checkpoint_dir = self._create_wandb_run(
            custom_run_id=custom_run_id
        )
        print(os.path.join(self.run_checkpoint_dir, "checkpoint.pth"))
        # if self.wandb_run.resumed or self.config.train.resume_from_checkpoint:
        if os.path.exists(os.path.join(self.run_checkpoint_dir, "checkpoint.pth")):
            print("Resuming from checkpoint")
            self._load_checkpoint()
            print(f"Checkpoint loaded at epoch {self.epoch}")
        else:
            self.epoch = 0
            print("No checkpoint found, starting from scratch.")

        if hasattr(self, "scheduler") and self.scheduler is not None:
            for _ in range(self.epoch):
                self.scheduler.step()

        self._train_loop()

        print("Training complete")

        self._save_checkpoint()

        if hasattr(self.config, "retrain") and self.config.retrain:
            self._retrain()

        self._log_after_train()

        wandb.finish()

    def _make_reproducible(self):
        set_seed(self.config.train.seed)

        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    def _create_datasets(self):
        return get_dataset(self.config.data.name, self.config)

    def _create_dataloaders(self):
        # get the number of workers
        num_workers = 4 if DEVICE == "cuda" else 0
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.data.train_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.data.val_batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        return train_loader, val_loader

    def _create_model(self):
        model = get_model(self.config.model.name, self.config)
        print("GPU number: ", torch.cuda.device_count())
        if torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs")
            model = nn.DataParallel(model)  # wrap model with nn.DataParallel
        model.to(DEVICE)
        return model

    def _create_loss_fn(self):
        task = (
            "classification"
            if "classification" in self.config.task_id
            else "regression"
        )
        if task == "regression":
            return nn.MSELoss(reduction="mean")
        elif task == "classification":
            return nn.CrossEntropyLoss()

    def _create_optimizer(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.config.optim.lr,
            momentum=self.config.optim.kwargs.momentum,
            weight_decay=self.config.optim.kwargs.weight_decay,
        )
        if self.config.optim.decay_schedule == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config.train.epochs
            )
            return optimizer, scheduler
        return optimizer, None

    def _create_wandb_run(self, custom_run_id=None):
        if custom_run_id is not None:
            custom_run_id = custom_run_id
        elif "SLURM_JOB_ID" in os.environ:
            custom_run_id = os.environ["SLURM_JOB_ID"]
            print(f"SLURM_JOB_ID: {custom_run_id}")
            print(f"Custom run id: {custom_run_id}")
        else:
            custom_run_id = None

        run = wandb.init(
            project=self.config.wandb_project,
            entity=self.config.wandb_entity,
            dir=os.environ["WANDB_DIR"],
            id=custom_run_id,
            mode=self.config.wandb_mode,
            resume="allow",
        )

        wandb.config.update(self.config.to_dict(), allow_val_change=True)

        print(f"Initialized WandB run with id {run.id}")
        # NOTE: In offline mode, run.id is not set and is None thus has a different path

        run_checkpoint_dir = os.path.join(os.environ["CHECKPOINT_DIR"], run.id)
        print(f"Checkpoint dir: {run_checkpoint_dir}")
        return run, run_checkpoint_dir

    def _save_checkpoint(self):
        os.makedirs(self.run_checkpoint_dir, exist_ok=True)

        checkpoint = {
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(checkpoint, os.path.join(self.run_checkpoint_dir, "checkpoint.pth"))
        print("Checkpoint saved")

    def _load_checkpoint(self):
        checkpoint = torch.load(os.path.join(self.run_checkpoint_dir, "checkpoint.pth"))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint["epoch"]
        print("Checkpoint loaded")

    def __submitit_checkpoint__(self):
        """Function used by submitit when SLURM job is preempted"""
        resume_config = self.config.copy()
        with resume_config.unlocked():
            resume_config.train.resume_from_checkpoint = True
        resume_trainer = Trainer(resume_config)
        return submitit.helpers.DelayedSubmission(resume_trainer)

    def _train_loop(self):
        print(self.config)
        while self.epoch < self.config.train.epochs:
            self.epoch += 1
            if (
                self.epoch == self.config.train.epochs
                and hasattr(self.config, "retrain")
                and self.config.retrain
            ):
                return_ws_list = True
            else:
                return_ws_list = False

            set_seed(1000 * (self.epoch + 1) + self.config.train.seed)
            start_epoch_time = time.time()
            if (
                hasattr(self.config.optim.regularization, "rho_schedule")
                and self.config.optim.regularization.rho_schedule
            ):
                self.config.optim.regularization.rho = rho_scheduler(
                    epoch=self.epoch, max_epochs=self.config.train.epochs
                )

            loss, accuracy, metrics, ws_lists = train_epoch(
                model=self.model,
                optimizer=self.optimizer,
                data_loader=self.train_loader,
                loss_fn=self.loss_fn,
                config=self.config.optim,
                device=self.device,
                return_ws_list=return_ws_list,
            )

            if hasattr(self, "scheduler") and self.scheduler is not None:
                self.scheduler.step()

            print(
                f"Epoch {self.epoch} took {time.time() - start_epoch_time:.2f} seconds"
            )

            if self.config.train.checkpoint:
                self._save_checkpoint()

            self._log_after_epoch(loss=loss, train_accuracy=accuracy, metrics=metrics)

        # save the dict ws_lists
        if hasattr(self.config, "retrain") and self.config.retrain:
            torch.save(ws_lists, os.path.join(self.run_checkpoint_dir, "ws_lists.pth"))

    def _log_after_epoch(self, loss, train_accuracy, metrics):
        if (
            self.config.verbose.verbose
            and self.epoch % self.config.verbose.log_freq == 0
        ):
            print(f"Epoch: {self.epoch}, Loss: {loss}")
            wandb.log({"Training Loss": loss}, step=self.epoch)

            print(f"Train Accuracy: {train_accuracy}")
            wandb.log({"Train Accuracy": train_accuracy}, step=self.epoch)

            val_accuracy = calculate_accuracy(self.model, self.val_loader, DEVICE)
            print(f"Validation Accuracy: {val_accuracy}")
            wandb.log({"Validation Accuracy": val_accuracy}, step=self.epoch)

            # log sparsity and distince non-zero weights
            sparsity = compute_sparsity(
                self.model, self.config.optim.regularization.prox_layer
            )
            wandb.log({"Sparsity": sparsity * 100}, step=self.epoch)

            distinct_w = compute_distinct_non_zero_weights(
                self.model, self.config.optim.regularization.prox_layer
            )
            wandb.log({"Distinct Non-zero Weights": distinct_w * 100}, step=self.epoch)

            if len(metrics) == 0:
                sparsity = compute_sparsity(self.model)
                print(f"Sparsity Weight Layer_1: {sparsity * 100}%")
                wandb.log({"Sparsity Weight Layer_1": sparsity * 100}, step=self.epoch)
            else:
                for m in metrics:
                    print(f"Metrics {m}: {metrics[m]}")
                    distinct_w, sparsity = metrics[m]

                    print(f"Distinct {m}: {distinct_w * 100}")
                    wandb.log({f"Distinct {m}": distinct_w * 100}, step=self.epoch)
                    print(f"Sparsity {m}: {sparsity * 100}%")
                    wandb.log({f"Sparsity {m}": sparsity * 100}, step=self.epoch)
        if (
            self.config.verbose.verbose
            and self.epoch % (10 * self.config.verbose.log_freq) == 0
        ):
            # save the heatmap of the first layer weights
            map_shape = self.config.data.input_shape
            if (
                self.config.model.name == "SkinnyCNN"
                or self.config.model.name == "SkinnyCNNPrime_FC"
                or self.config.model.name == "AlexNet"
                or self.config.model.name == "AlexNetPrime_FC"
            ):
                map_shape = (
                    map_shape[0],
                    map_shape[1] + 2 * 2,
                    map_shape[2] + 2 * 2,
                )  # add padding

            fig = get_heatmap_fig(model=self.model, map_shape=map_shape)
            wandb.log({f"heatmap_{self.epoch}": fig}, step=self.epoch)

    def _log_after_train(self):
        # save final checkpoint in wandb
        wandb.save(os.path.join(self.run_checkpoint_dir, "checkpoint.pth"))

    def _retrain(self):
        print("Retraining")
        ws_lists = torch.load(os.path.join(self.run_checkpoint_dir, "ws_lists.pth"))
        self.config.optim.regularization.ws_lr = (0.0, 0.0)
        self.config.optim.regularization.lasso_lr = 0.0
        self.config.optim.regularization.rho = 0.0
        retrain_epoch = 0
        self.optimizer, self.scheduler = self._create_optimizer()
        while retrain_epoch < self.config.train.epochs:
            retrain_epoch += 1
            self.epoch += 1
            set_seed(1000 * (self.epoch + 1) + self.config.train.seed)
            start_epoch_time = time.time()

            loss, accuracy, metrics, _ = train_epoch(
                model=self.model,
                optimizer=self.optimizer,
                data_loader=self.train_loader,
                loss_fn=self.loss_fn,
                config=self.config.optim,
                device=self.device,
                retrain_ws_list=ws_lists,
            )

            if hasattr(self, "scheduler") and self.scheduler is not None:
                self.scheduler.step()

            print(
                f"Epoch {self.epoch} took {time.time() - start_epoch_time:.2f} seconds"
            )

            if self.config.train.checkpoint:
                self._save_checkpoint()

            self._log_after_epoch(loss=loss, train_accuracy=accuracy, metrics=metrics)

        print("Retraining done")


def main(_):
    config = FLAGS.config
    # trainer(config)
    trainer = Trainer(config)

    if config.resource.cluster == "debug":
        trainer.__call__()
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    else:
        job_submitit_dir = os.environ["SUBMITIT_DIR"]

        executer = submitit.AutoExecutor(
            folder=job_submitit_dir,
        )
        if config.resource.cluster == "Mila":
            executer.update_parameters(
                name=config.data.name,
                slurm_partition=config.resource.partition,
                slurm_mem=config.resource.mem,
                slurm_time=config.resource.time,
                nodes=config.resource.nodes,
                tasks_per_node=config.resource.tasks_per_node,
                cpus_per_task=config.resource.cpus_per_task,
                gpus_per_node=config.resource.gpus_per_node,
                slurm_exclude=DEFAULT_EXCLUDE,
            )
        elif config.resource.cluster == "ComputeCanada":
            executer.update_parameters(
                name=config.data.name,
                slurm_mem=config.resource.mem,
                slurm_time=config.resource.time,
                nodes=config.resource.nodes,
                tasks_per_node=config.resource.tasks_per_node,
                cpus_per_task=config.resource.cpus_per_task,
                gpus_per_node=config.resource.gpus_per_node,
            )

        job = executer.submit(trainer)
        print(job.job_id)


if __name__ == "__main__":
    app.run(main)
