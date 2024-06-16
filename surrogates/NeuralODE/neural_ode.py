import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchdiffeq import odeint, odeint_adjoint

from surrogates.surrogates import AbstractSurrogateModel
from surrogates.NeuralODE.neural_ode_config import NeuralODEConfigOSU as Config


class NeuralODE(AbstractSurrogateModel):

    def __init__(self, config: Config = Config()):
        super().__init__()
        self.model = ModelWrapper(config=config).to(config.device)
        self.train_loss = None

    def forward(self, x):
        t_range = self._get_t_range()
        return self.model(x, t_range)

    def prepare_data(self, data_loader):
        pass

    def fit(self, conf, data_loader, test_loader=None):
        # TODO: make Optimizer and scheduler configable
        optimizer = Adam(self.model.parameters(), lr=self.config.learnign_rate)
        scheduler = None
        if self.config.final_learning_rate is not None:
            scheduler = CosineAnnealingLR(
                optimizer, self.config.epochs, eta_min=self.config.final_learning_rate
            )
        t_range = self._get_t_range()
        for epoch in range(self.config.epochs):
            for i, x_true in enumerate(data_loader):
                optimizer.zero_grad()
                x0 = x_true[:, :, 0]
                x_pred = self.model(x0, t_range)
                loss = self.model.total_loss(x_true, x_pred)
                loss.backward()
                optimizer.step()
            if scheduler is not None:
                scheduler.step()

    def predict(self, data_loader):
        pass

    def save(
        self, model_name, config, subfolder, train_loss, test_loss, training_duration
    ):
        pass

    def _get_t_range(self):
        return torch.linspace(0, 1, self.config.t_steps)


class ModelWrapper(torch.nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.encoder = Encoder(
            in_features=config.in_features,
            latent_features=config.latent_features,
            n_hidden=config.coder_hidden,
            width_list=config.coder_layers,
            activation=config.coder_activation,
        )
        self.decoder = Decoder(
            out_features=config.in_features,
            latent_features=config.latent_features,
            n_hidden=config.coder_hidden,
            width_list=config.coder_layers,
            activation=config.coder_activation,
        )
        self.ode = ODE(
            input_shape=config.in_features,
            output_shape=config.in_features,
            activation=config.ode_activation,
            n_hidden=config.ode_hidden,
            layer_width=config.ode_layer_width,
            tanh_reg=config.ode_tanh_reg,
        )

    def forward(self, x0, t_range):
        z0 = self.encoder(x0)  # x(t=0)
        if self.config.use_adjoint:
            result = odeint_adjoint(
                func=self.ode,
                y0=z0,
                t=t_range,
                adjoint_rtol=self.config.rtol,
                adjoint_atol=self.config.atol,
                adjoint_method=self.config.method,
            )
            if not isinstance(result, torch.Tensor):
                raise TypeError("odeint_adjoint must return tensor, check inputs")
            return self.decoder(torch.permute(result, dims=(1, 0, 2)))
        result = odeint(
            func=self.ode,
            y0=z0,
            t=t_range,
            rtol=self.config.rtol,
            atol=self.config.atol,
            method=self.config.method,
        )
        if not isinstance(result, torch.Tensor):
            raise TypeError("odeint must return tensor, check inputs")
        return self.decoder(torch.permute(result, dims=(1, 0, 2)))


class ODE(torch.nn.Module):

    def __init__(
        self,
        input_shape: int,
        output_shape: int,
        activation: torch.nn.Module,
        n_hidden: int,
        layer_width: int,
        tanh_reg: bool,
    ):
        super().__init__()

        self.tanh_reg = tanh_reg
        self.reg_factor = torch.nn.Parameter(torch.tensor(1.0))
        self.activation = activation

        self.mlp = torch.nn.Sequential()
        self.mlp.append(torch.nn.Linear(input_shape, layer_width))
        self.mlp.append(self.activation)
        for i in range(n_hidden):
            self.mlp.append(torch.nn.Linear(layer_width, layer_width))
            self.mlp.append(self.activation)
        self.mlp.append(torch.nn.Linear(layer_width, output_shape))

    def forward(self, t, x):
        if self.tanh_reg:
            return self.reg_factor * torch.tanh(self.mlp(x) / self.reg_factor)
        return self.mlp(x)


class Encoder(torch.nn.Module):

    def __init__(
        self,
        in_features: int = 29,
        latent_features: int = 5,
        n_hidden: int = 4,
        width_list: list = [32, 16, 8],
        activation: torch.nn.Module = torch.nn.ReLU(),
    ):
        super().__init__()
        assert (
            n_hidden == len(width_list) + 1
        ), "n_hidden must equal length of width_list"
        self.in_features = in_features
        self.latent_features = latent_features
        self.n_hidden = n_hidden
        self.width_list = width_list
        self.activation = activation

        self.mlp = torch.nn.Sequential()
        self.mlp.append(torch.nn.Linear(self.in_features, self.width_list[0]))
        self.mlp.append(self.activation)
        for i, width in enumerate(self.width_list[1:]):
            self.mlp.append(torch.nn.Linear(self.width_list[i], width))
            self.mlp.append(self.activation)
        self.mlp.append(torch.nn.Linear(self.width_list[-1], self.latent_features))
        self.mlp.append(torch.nn.Tanh())

    def forward(self, x):
        return self.mlp(x)


class Decoder(torch.nn.Module):

    def __init__(
        self,
        out_features: int,
        latent_features: int,
        n_hidden: int,
        width_list: list,
        activation: torch.nn.Module,
    ):
        super().__init__()
        assert (
            n_hidden == len(width_list) + 1
        ), "n_hidden must equal length of width_list"
        self.out_features = out_features
        self.latent_features = latent_features
        self.n_hidden = n_hidden
        self.width_list = width_list
        self.activation = activation
        self.width_list.reverse()

        self.mlp = torch.nn.Sequential()
        self.mlp.append(torch.nn.Linear(self.latent_features, self.width_list[0]))
        self.mlp.append(self.activation)
        for i, width in enumerate(self.width_list[1:]):
            self.mlp.append(torch.nn.Linear(self.width_list[i], width))
            self.mlp.append(self.activation)
        self.mlp.append(torch.nn.Linear(self.width_list[-1], self.out_features))
        self.mlp.append(torch.nn.Tanh())

    def forward(self, x):
        return self.mlp(x)
