import torch

# from torch.autograd.functional import jacobian
from torchdiffeq import odeint_adjoint
from params import DEVICE
from utilities import mass_function, relative_error, deriv, deriv2
from functorch import vmap, jacrev, jacfwd  # , hessian


class NeuralODE(torch.nn.Module):

    def __init__(
        self,
        input_shape: int,
        output_shape: int,
        activation: torch.nn.Module = torch.nn.Tanh(),
        n_hidden: int = 8,
        layer_width: int = 256,
        tanh_reg=False,
        show: bool = True,
    ):
        super().__init__()

        self.tanh_reg = tanh_reg
        self.reg_factor = torch.nn.Parameter(torch.Tensor([1.0]))

        self.input_shape = input_shape
        self.activation = activation

        self.mlp = torch.nn.Sequential()
        self.mlp.append(torch.nn.Linear(self.input_shape, layer_width))
        self.mlp.append(self.activation)
        for i in range(n_hidden):
            self.mlp.append(torch.nn.Linear(layer_width, layer_width))
            self.mlp.append(self.activation)
        self.mlp.append(torch.nn.Linear(layer_width, output_shape))
        if show:
            print(self.mlp)

    def forward(self, t, x):  # t input for odeint compatibility
        if self.tanh_reg:
            return self.reg_factor * torch.tanh(self.mlp(x) / self.reg_factor)
        else:
            return self.mlp(x)

    def augmented_forward(
        self, x
    ):  # same as forward but without t input (for calculating derivatives)
        if self.tanh_reg:
            return self.reg_factor * torch.tanh(self.mlp(x) / self.reg_factor)
        else:
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
        out_features: int = 29,
        latent_features: int = 5,
        n_hidden: int = 4,
        width_list: list = [32, 16, 8],
        activation: torch.nn.Module = torch.nn.ReLU(),
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


# wrap models in nn.torch.nn.Module child for autograd
class ModelWrapper(torch.nn.Module):
    def __init__(
        self,
        real_vars: int = 29,
        latent_vars: int = 5,
        ode_width: int = 256,
        ode_hidden: int = 8,
        tanh_reg=False,  # tanh regularzation of form q * tanh(mlp(x) / q)
        coder_hidden: int = 4,
        width_list: list = [32, 16, 8],
        coder_activation: torch.nn.Module = torch.nn.ReLU(),
        losses_list: list = ["L2", "id", "deriv", "deriv2"],
        loss_weights: torch.Tensor = torch.tensor([1, 1, 1, 1]),
    ):
        super().__init__()
        assert (
            coder_hidden == len(width_list) + 1
        ), "coder_hidden must equal length of width_list"
        self.x_vars = real_vars
        self.z_vars = latent_vars
        self.encoder = Encoder(
            self.x_vars,
            self.z_vars,
            n_hidden=coder_hidden,
            width_list=width_list,
            activation=coder_activation,
        )
        self.decoder = Decoder(
            self.x_vars,
            self.z_vars,
            n_hidden=coder_hidden,
            width_list=width_list,
            activation=coder_activation,
        )
        self.ode = NeuralODE(
            self.z_vars,
            self.z_vars,
            n_hidden=ode_hidden,
            layer_width=ode_width,
            tanh_reg=tanh_reg,
            show=False,
        )
        self.width = ode_width
        self.hidden = ode_hidden
        self.tanh_reg = tanh_reg
        self.losses_list = losses_list
        self.loss_weights = loss_weights

        # cached variables
        self.z_pred = None
        self.x_pred = None
        self.z_dot = None
        self.relative_error = None

    def forward(self, x0: torch.Tensor, t_range: torch.Tensor) -> torch.Tensor:
        self.z_pred = torch.permute(
            odeint_adjoint(
                self.ode,
                self.encoder(x0),
                t_range,
                adjoint_rtol=1e-7,
                adjoint_atol=1e-9,
                adjoint_method="dopri8",
            ),
            (1, 0, 2),
        )
        self.z_dot = torch.gradient(self.z_pred, dim=1)[0]
        self.x_pred = self.decoder(self.z_pred)
        return self.x_pred

    def encoder_loss(self, x: torch.Tensor, x_dot: torch.Tensor):  # L1 term in T.Grassi
        xdot_grad_enc = torch.autograd.functional.jvp(self.encoder, x, x_dot)[1]
        return self.l2_loss(self.z_pred, xdot_grad_enc)
        # jac_enc = jacobian(self.encoder, x, vectorize=True)
        # return self.l2_loss(self.z_pred, torch.einsum('lmnijk,ijk->lmn', jac_enc, x_dot))#torch.matmul(jac_enc.flatten(start_dim=3, end_dim=5), x_dot.flatten()))

    def decoder_loss(self, x_dot: torch.Tensor):  # L2 term in T.Grassi
        zdot_grad_dec = torch.autograd.functional.jvp(
            self.decoder, self.z_pred, self.z_dot
        )[1]
        return self.l2_loss(x_dot, zdot_grad_dec)
        # jac_dec = jacobian(self.decoder, self.z_pred, vectorize=True, strategy='forward-mode') # forward mode is more performant for more outputs than inputs
        # return self.l2_loss(x_dot, torch.einsum('lmnijk,ijk->lmn', jac_dec, self.z_pred))

    def identity_loss(self, x: torch.Tensor):
        return self.l2_loss(x, self.decoder(self.encoder(x)))

    def ode_batch_hessian_loss(self, x, reverse_mode=True):
        if reverse_mode:
            return torch.abs(
                vmap(vmap(jacfwd(jacrev(self.ode.augmented_forward))))(x)
            ).sum()
        else:
            return torch.abs(
                vmap(vmap(jacrev(jacfwd(self.ode.augmented_forward))))(x)
            ).sum()

    def mass_conservation_loss(self, x_true, x_pred):
        # return torch.abs(vmap(vmap(mass_function))(x_true) - vmap(vmap(mass_function))(x_pred)).mean()
        return torch.sum(
            torch.abs(mass_function(x_true) - mass_function(x_pred)), dim=-1
        ).mean()
        # zu jedem zeitpunk masse berechnen, differenz, absolut, summieren, mean über batch

    def relative_l2_loss(self, x_true, x_pred):
        # cache relative error to avoid recomputing
        self.relative_error = relative_error(x_true, x_pred)
        return self.relative_error.mean()

    def total_loss(
        self,
        x_true: torch.Tensor,
        x_pred: torch.Tensor,
    ) -> torch.Tensor:
        use_losses = self.losses_list
        weights = self.loss_weights
        losses = torch.empty(len(use_losses)).to(DEVICE)
        for i, (loss_type, weight) in enumerate(zip(use_losses, weights)):
            if loss_type == "L2":
                loss = weight * self.l2_loss(x_true, x_pred)
                losses[i] = loss
            elif loss_type == "relL2":
                loss = weight * self.relative_l2_loss(x_true, x_pred)
                losses[i] = loss
            elif loss_type == "id":
                loss = weight * self.identity_loss(x_true)
                losses[i] = loss
            elif loss_type == "hess":
                loss = weight * self.ode_batch_hessian_loss(self.z_pred)
                losses[i] = loss
            elif loss_type == "mass":
                loss = weight * self.mass_conservation_loss(x_true, x_pred)
                losses[i] = loss
            elif loss_type == "deriv2":
                loss = weight * self.l2_loss(deriv2(x_pred), deriv2(x_true))
                losses[i] = loss
            elif loss_type == "deriv":
                loss = weight * self.l2_loss(deriv(x_pred), deriv(x_true))
                losses[i] = loss
        return losses

    @classmethod
    def l2_loss(cls, y1: torch.Tensor, y2: torch.Tensor):
        return torch.mean(torch.abs(y2 - y1) ** 2)


class LinearODE(torch.nn.Module):

    def __init__(self, latent_params: int = 5):
        super().__init__()
        self.slope = torch.nn.Parameter(torch.randn(latent_params))
        # self.bias = torch.nn.Parameter(torch.randn(latent_params))
        self.latent_params = latent_params

    def forward(self, t):
        self.batch_size = t.shape[0]
        return torch.einsum("bi,j->bij", (t, self.slope))
        # return self.bias

    def augmented_forward(self, t):
        ones = torch.ones(self.batch_size).to(DEVICE)
        return torch.einsum("b,i,j->bij", (ones, t, self.slope))


class LinearModelWrapper(ModelWrapper):
    def __init__(
        self,
        real_vars: int = 29,
        latent_vars: int = 5,
        coder_hidden: int = 4,
        width_list: list = [32, 16, 8],
        coder_activation: torch.nn.Module = torch.nn.ReLU(),
    ):
        super().__init__(
            real_vars,
            latent_vars,
            1,
            1,
            None,
            coder_hidden,
            width_list,
            coder_activation,
        )
        self.ode = LinearODE(latent_vars)

    def forward(self, x0: torch.Tensor, t_range: torch.Tensor) -> torch.Tensor:
        batch_size = x0.shape[0]
        t = t_range.unsqueeze(0).repeat(batch_size, 1)
        self.z_pred = self.ode(t) + self.encoder(x0).unsqueeze(1)
        self.z_dot = self.ode.slope
        self.x_pred = self.decoder(self.z_pred)
        return self.x_pred


class PolynomialODE(torch.nn.Module):

    def __init__(self, degree: int = 2, latent_params: int = 5):
        super().__init__()
        self.coef = torch.nn.Linear(
            degree, latent_params, bias=False, dtype=torch.float32
        )
        self.degree = degree

    def forward(self, t):
        return self.coef(self.poly(t))

    def poly(self, t):
        t = t.unsqueeze(1)
        return torch.hstack([t**i for i in range(1, self.degree + 1)]).permute(0, 2, 1)

    def integrated_poly(self, t):
        t = t.unsqueeze(1)
        return torch.hstack(
            [t ** (i + 1) / (i + 1) for i in range(self.degree)]
        ).permute(0, 2, 1)


class PolynomialModelWrapper(ModelWrapper):
    def __init__(
        self,
        real_vars: int = 29,
        latent_vars: int = 5,
        degree: int = 2,
        coder_hidden: int = 4,
        width_list: list = [32, 16, 8],
        coder_activation: torch.nn.Module = torch.nn.ReLU(),
    ):
        super().__init__(
            real_vars,
            latent_vars,
            1,
            1,
            None,
            coder_hidden,
            width_list,
            coder_activation,
        )
        self.ode = PolynomialODE(degree, latent_vars)

    def forward(self, x0: torch.Tensor, t_range: torch.Tensor) -> torch.Tensor:
        batch_size = x0.shape[0]
        t = t_range.unsqueeze(0).repeat(batch_size, 1)
        self.z_pred = self.ode(t) + self.encoder(x0).unsqueeze(1)
        # self.z_dot = self.ode.slope
        self.x_pred = self.decoder(self.z_pred)
        return self.x_pred
