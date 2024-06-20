import numpy as np
import torch
from torch.utils.data import Dataset

# from params import REAL_VARS, MODELS_FOLDER, PLOT_FOLDER, DEVICE, MASSES
# from params import DEVICE
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

# from datetime import datetime
# import pandas as pd
# import os
from functorch import vmap, jacrev, jacfwd  # , hessian
from matplotlib.offsetbox import AnchoredText

# import h5py


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_data_from_files():
    dataset = torch.empty((500, 100, 58))
    for i in range(500):
        num = (4 - len(str(i))) * "0" + str(i)
        # /Users/immi/Documents/python/Bachelorarbeit_Material/tgrassi-latent_ode_paper/chemistry/outputs/chemistry_
        # /export/home/isulzer/tgrassi-latent_ode_paper/chemistry/outputs/chemistry_
        dataset[i, :, :] = torch.Tensor(
            np.loadtxt(
                "/export/home/isulzer/tgrassi-latent_ode_paper/chemistry/outputs/chemistry_"
                + num
                + ".dat"
            )
        )
    return dataset


def load_test_data():
    dataset = torch.empty((50, 100, 58))
    for i in range(50):
        num = (4 - len(str(i))) * "0" + str(i)
        dataset[i, :, :] = torch.Tensor(
            np.loadtxt(
                "/export/home/isulzer/tgrassi-latent_ode_paper/chemistry/test_outputs/chemistry_"
                + num
                + ".dat"
            )
        )
    return dataset


# def make_run_directories(id):
#     model_subfolder = MODELS_FOLDER + f"run_id_{id}/"
#     if not os.path.exists(model_subfolder):
#         os.makedirs(model_subfolder)

#     plot_subfolder = PLOT_FOLDER + f"run_id_{id}/"
#     if not os.path.exists(plot_subfolder):
#         os.makedirs(plot_subfolder)

#     return model_subfolder, plot_subfolder


def plot_evolution(filepath, x_true, x_pred, t):
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    for i in range(x_true.shape[-1]):
        ax[0].plot(t, x_true[:, i])
        ax[0].set_title("True evolution")
        ax[0].set_xlabel("time")
        ax[1].plot(t, x_pred[:, i])
        ax[1].set_title("Predicted evolution")
        ax[1].set_xlabel("time")
    fig.suptitle("Real space evolution")
    plt.savefig(filepath)
    plt.close()


def plot_scatter(filepath, x_true, x_pred):
    plt.figure(figsize=(6, 6))
    for i in range(x_true.shape[-1]):
        plt.scatter(x_true, x_pred, s=1)
    at = AnchoredText(
        f"RMSE = {round(rmse(x_true, x_pred).item(), 3)}",
        prop=dict(size=15),
        frameon=True,
        loc="upper left",
    )
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    plt.axes().add_artist(at)
    plt.plot([-1, 1], [-1, 1], alpha=0.3, color="black")
    plt.xlim(-1e0, 1e0)
    plt.ylim(-1e0, 1e0)
    plt.xlabel("true abundance")
    plt.ylabel("predicted abundance")
    plt.savefig(filepath)
    plt.close()


def plot_latent(filepath, z_true, z_pred, t):
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    for i in range(z_true.shape[-1]):
        ax[0].plot(t, z_true[:, i])
        ax[0].set_title("encoded evolution")
        ax[0].set_xlabel("time")
        ax[1].plot(t, z_pred[:, i])
        ax[1].set_title("Predicted evolution")
        ax[1].set_xlabel("time")
    fig.suptitle("Latent evolution")
    plt.savefig(filepath)
    plt.close()


def plot_loss(filepath, losses: torch.Tensor, loss_list):
    # fig, ax = plt.subplots(1, len(loss_list), figsize=(13, 6))
    # norm = losses.max()
    x_range = torch.linspace(0, losses.shape[0], losses.shape[0])
    plt.figure()
    plt.yscale("log")
    for i, loss_name in enumerate(loss_list):
        plt.plot(x_range, losses[:, i], label=loss_name, alpha=0.5)
        plt.xlabel("epoch")
        plt.ylabel("normalized loss")
        plt.title("Loss history")
    plt.legend()
    plt.savefig(filepath)
    plt.close()


def plot_relative_error(filepath, rel_error: torch.Tensor):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].set_yscale("log")
    x_range = torch.linspace(0, rel_error.shape[0], rel_error.shape[0])
    for i in range(rel_error.shape[-1]):
        ax[0].plot(x_range, rel_error[:, i], alpha=0.5)
    ax[0].set_xlabel("time")
    ax[0].set_ylabel("relative error")
    ax[0].set_title("Relative error in time per species")
    _, bins = np.histogram(rel_error.flatten(), bins=20)
    logbins = np.logspace(
        np.max((np.log10(bins[0]), np.log10(10e-8))), np.log10(bins[-1]), len(bins)
    )
    ax[1].hist(rel_error.flatten(), bins=logbins)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("order of magnitude")
    ax[1].set_ylabel("count")
    ax[1].set_title("Relative error over all data points")
    at = AnchoredText(
        f"std. dev. = {torch.std(rel_error):.3E}",
        prop=dict(size=10),
        frameon=True,
        loc="upper left",
    )
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax[1].add_artist(at)
    ax[2].plot(x_range, rel_error.mean(dim=-1))
    ax[2].set_xlabel("time")
    ax[2].set_ylabel("relative error")
    ax[2].set_title("Mean relative error")
    ax[2].set_yscale("log")
    at = AnchoredText(
        f"mean = {rel_error.mean():.3E}",
        prop=dict(size=10),
        frameon=True,
        loc="upper right",
    )
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax[2].add_artist(at)
    fig.savefig(filepath)
    plt.close()


def nth_derivative(f: torch.Tensor, at: torch.Tensor, order: int):
    gradient = torch.autograd.grad(f, at, create_graph=True, retain_graph=True)[0]
    f = gradient.sum()
    derivatives = gradient[None]
    for i in range(order - 1):
        gradient = torch.autograd.grad(f, at, create_graph=True, retain_graph=True)[0]
        f = gradient.sum()
        derivatives = torch.vstack(
            (derivatives, gradient[None]),
        )
    return derivatives


def batch_jacobian(f, x, reverse_mode=True):
    if reverse_mode:
        return vmap(vmap(jacrev(f)))(x)
    else:
        return vmap(vmap(jacfwd(f)))(x)


def batch_hessian(f, x, reverse_mode=True):
    # return vmap(vmap(hessian(f)))(x)
    if reverse_mode:
        return vmap(vmap(jacfwd(jacrev(f))))(x)
    else:
        return vmap(vmap(jacrev(jacfwd(f))))(x)


def batch_3rd(f, x):
    return (vmap(vmap(jacfwd(jacrev(jacfwd(f))))))(x)
    # return(vmap(vmap(jacrev(jacfwd(jacrev(f))))))(x)


def jac_det(f, x):
    return vmap(vmap(jacrev(f)))(x).sum()


# def mass_function(n: torch.Tensor):
#     return torch.sum(MASSES * n, dim=-1)


def mass_function(n: torch.Tensor):
    raise NotImplementedError("mass function not implemented, please don't use it")


def relative_error(x_true: torch.Tensor, x_pred: torch.Tensor):
    return torch.abs((x_pred - x_true) / torch.amax(x_true, dim=1)[:, None, :])
    # return torch.mean(torch.abs(x_pred - x_true) / torch.abs(x_true))


def rmse(x_true: torch.Tensor, x_pred: torch.Tensor):
    return torch.sqrt(torch.mean((x_pred - x_true) ** 2, dim=(0, 1)))


def deriv(x):
    return torch.gradient(x, dim=1)[0].squeeze(0)


def deriv2(x):
    return deriv(deriv(x))


class ChemDataset(torch.utils.data.Dataset):

    def __init__(self, raw_data, device):
        self.data = torch.tensor(raw_data, dtype=torch.float64)
        self.xmin = self.data.min()
        self.xmax = self.data.max()
        self.data = 2 * (self.data - self.xmin) / (self.xmax - self.xmin) - 1
        self.length = self.data.shape[0]
        if not self.data.dtype == torch.float64:
            self.data = torch.tensor(self.data, dtype=torch.float64)
        self.data = self.data.to(device)

    def __getitem__(self, index):
        return self.data[index, :, :]

    # def __getitems__(self, index_list: list[int]):
    #     return self.data[index_list, :, :]

    def __len__(self):
        return self.length
