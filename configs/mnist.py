import ml_collections as mlc


def get_config():
    config = mlc.ConfigDict()

    config.task_id = "classification_mnist"

    config.data = mlc.ConfigDict()
    config.data.name = "MNIST"
    config.data.input_shape = (1, 28, 28)
    config.data.output_dim = 10
    config.data.train_batch_size = 512
    config.data.val_batch_size = 512

    config.train = mlc.ConfigDict()
    config.train.epochs = 10
    config.train.seed = 123456789
    config.train.checkpoint = False
    config.train.resume_from_checkpoint = False

    config.model = mlc.ConfigDict()
    config.model.name = "FullyConnectedMNIST"
    config.model.num_hidden_layers = 1
    config.model.kwargs = mlc.ConfigDict()
    config.model.kwargs.first_out_channel = 150

    config.optim = mlc.ConfigDict()
    config.optim.name = "SGD"
    config.optim.lr = 1e-2
    config.optim.kwargs = mlc.ConfigDict()
    config.optim.kwargs.momentum = 0.0
    config.optim.kwargs.weight_decay = 0.0
    config.optim.decay_schedule = "None"
    config.optim.regularization = mlc.ConfigDict()
    config.optim.regularization.ws_lr = 0.0
    config.optim.regularization.ws_alg = "search_collisions"
    config.optim.regularization.lasso_lr = 0.0
    config.optim.regularization.use_trick = False
    config.optim.regularization.rho = 0.0
    config.optim.regularization.prox_layer = "FC1"  # or "FC12" "All"
    config.optim.regularization.beta = 0.0
    config.optim.regularization.lmbda = 0.0

    config.verbose = mlc.ConfigDict()
    config.verbose.verbose = True
    config.verbose.log_freq = 1  # Log every epoch for a task like MNIST
    config.wandb_mode = "disabled"
    config.wandb_entity = "motahareh-s"
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

    config.run_checkpoint_dir = "None"

    return config
