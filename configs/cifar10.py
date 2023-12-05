import ml_collections as mlc


def get_config():
    config = mlc.ConfigDict()

    config.task_id = "classification_cifar10"

    config.data = mlc.ConfigDict()
    config.data.name = "CIFAR10"
    config.data.input_shape = (3, 32, 32)  
    config.data.output_dim = 10
    config.data.train_batch_size = 512
    config.data.val_batch_size = 512

    config.train = mlc.ConfigDict()
    config.train.epochs = 30
    config.train.seed = 123456789
    config.train.checkpoint = False
    config.train.resume_from_checkpoint = False

    config.model = mlc.ConfigDict()
    config.model.name = "S_Conv"
    config.model.kwargs = mlc.ConfigDict()
    config.model.kwargs.first_out_channel = 150

    config.optim = mlc.ConfigDict()
    config.optim.name = "SGD"
    config.optim.lr = 5e-3
    config.optim.kwargs = mlc.ConfigDict()
    config.optim.kwargs.momentum = 0.0
    config.optim.kwargs.weight_decay = 0.0
    config.optim.decay_schedule = "None"
    config.optim.regularization = mlc.ConfigDict()
    config.optim.regularization.ws_lr = 0.0
    config.optim.regularization.ws_alg = "search_collisions"
    config.optim.regularization.lasso_lr = 0.0
    config.optim.regularization.rho = 0.0
    config.optim.regularization.use_trick = False
    config.optim.regularization.prox_layer = "FC1"  # or "All"
    config.optim.regularization.beta = 0.0
    config.optim.regularization.lmbda = 0.0

    config.verbose = mlc.ConfigDict()
    config.verbose.verbose = True
    config.verbose.log_freq = 1
    config.wandb_mode = "disabled"
    config.wandb_entity = "entity-name"
    config.wandb_project = "proximal-weight-sharing"

    config.resource = mlc.ConfigDict()
    config.resource.cluster = "slurm"
    config.resource.partition = "long"
    config.resource.time = "24:00:00"
    config.resource.nodes = 1
    config.resource.tasks_per_node = 1
    config.resource.cpus_per_task = 1
    config.resource.gpus_per_node = 1
    config.resource.mem = "24GB"

    # config.run_checkpoint_dir = None

    return config
