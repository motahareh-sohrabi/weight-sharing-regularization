import numpy as np
import torch


class LinearDataset:
    """
    Generates a linear regression dataset with Gaussian noise.
    """

    def __init__(
        self,
        input_dim: int = 2,
        output_dim: int = 1,
        noise_std: float = 0.1,
        data_seed: int = 0,
        sample_seed: int = 0,
        num_samples: int = 100,
        distribution: str = "normal",
        normalize: bool = True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.noise_std = noise_std
        self.data_seed = data_seed
        self.sample_seed = sample_seed
        self.num_samples = num_samples
        self.distribution = distribution
        self.normalize = normalize

        self.data_rng = np.random.default_rng(self.data_seed)
        self.sample_rng = np.random.default_rng(self.sample_seed)

        self.initialize_distribution_parameters(distribution=self.distribution)

        self.X, self.y = self._generate_samples(num_samples=self.num_samples)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)

    def initialize_distribution_parameters(self, distribution: str = "normal"):
        # Generate "true" weights once with provided data_seed
        # Note that the shape of w is (output_dim, input_dim)
        if distribution == "normal":
            self.w = torch.tensor(
                self.data_rng.standard_normal((self.output_dim, self.input_dim)),
                dtype=torch.float32,
            )

        elif distribution == "shared":
            p = 0.8
            num_total_param = self.input_dim * self.output_dim
            num_shared_param = int(p * num_total_param)

            # Generate shared parameters
            shared_params_list = self.data_rng.standard_normal((num_shared_param))

            # Create weight matrix with shared parameters
            self.w = torch.empty((self.output_dim, self.input_dim), dtype=torch.float32)
            for i in range(self.w.shape[0]):
                for j in range(self.w.shape[1]):
                    index = self.data_rng.integers(0, num_shared_param)
                    shared_param = shared_params_list[index]
                    self.w[i, j] = round(shared_param, 2)

        if self.normalize:
            self.w = self.w / torch.norm(self.w)

        assert self.w.shape == (self.output_dim, self.input_dim)

    def _generate_samples(self, num_samples: int = 100):
        X = torch.tensor(
            self.sample_rng.standard_normal((num_samples, self.input_dim)),
            dtype=torch.float32,
        )
        noise = torch.tensor(
            self.sample_rng.standard_normal((num_samples, self.output_dim)),
            dtype=torch.float32,
        )

        y = X @ self.w.T + self.noise_std * noise

        assert X.shape == (num_samples, self.input_dim)
        assert noise.shape == (num_samples, self.output_dim)
        assert self.w.shape == (self.output_dim, self.input_dim)
        assert y.shape == (num_samples, self.output_dim)

        return X, y
