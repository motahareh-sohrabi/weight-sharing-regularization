import ml_collections as mlc


def get_config():
    config = mlc.ConfigDict()

    config.task_id = "linear_regression_synthetic"

    config.data = mlc.ConfigDict()
    config.data.name = "LinearDataset"
    config.data.data_seed = 12345
    config.data.sample_seed = 12345678
    config.data.input_dim = 100
    config.data.output_dim = 10
    config.data.noise_std = 0.1
    config.data.distribution = "normal"
    config.data.normalize = False
    config.data.train_samples = 1000
    config.data.train_batch_size = 128
    config.data.val_samples = 200
    config.data.val_batch_size = 128

    config.train = mlc.ConfigDict()
    config.train.epochs = 2000
    config.train.seed = 123456789

    config.model = mlc.ConfigDict()
    config.model.name = "LinearRegression"
    config.model.gamma = 0.0

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
    config.optim.regularization.rho = 0.0
    config.optim.regularization.prox_layer = "FC1"  # or "All"
    config.optim.regularization.beta = 0.0
    config.optim.regularization.lmbda = 0.0

    config.verbose = mlc.ConfigDict()
    config.verbose.verbose = False
    config.verbose.log_freq = 100
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

    return config
