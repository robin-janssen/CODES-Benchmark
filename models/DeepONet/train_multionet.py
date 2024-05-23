from __future__ import annotations
from time import time

import numpy as np
from tqdm import tqdm

# import optuna
import dataclasses

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from DeepONet.deeponet import MultiONet, MultiONetB, MultiONetT, OperatorNetworkType

from DeepONet.utils import get_project_path


from .train_utils import (
    time_execution,
    setup_optimizer_and_scheduler,
    setup_criterion,
    setup_losses,
    training_step,
)


def load_multionet(
    conf: type[dataclasses.dataclass] | dict,
    device: str = "cpu",
    model_path: str | None = None,
) -> OperatorNetworkType | tuple:
    """
    Load a MultiONet model from a saved state dictionary.
    If the path_to_state_dict is None, the function will return a new MultiONet model.

    Args:
        conf (type[dataclass]): A dataclass object containing the training configuration. It should have the following attributes:
            - 'pretrained_model_path' (str): Path to the saved state dictionary.
            - 'branch_input_size' (int): Input size for the branch network.
            - 'trunk_input_size' (int): Input size for the trunk network.
            - 'hidden_size' (int): Number of hidden units in each layer.
            - 'branch_hidden_layers' (int): Number of hidden layers in the branch network.
            - 'trunk_hidden_layers' (int): Number of hidden layers in the trunk network.
            - 'output_neurons' (int): Number of neurons in the last layer.
            - 'N_outputs' (int): Number of outputs.
            - 'architecture' (str): Architecture type, e.g., 'both', 'branch', or 'trunk'.
            - 'device' (str): The device to use for the model, e.g., 'cpu', 'cuda:0'.
        device (str): The device to use for the model.
        model_path (str): Path to the saved state dictionary. As seen from the parent directory of src.
    Returns:
        deeponet: Loaded DeepONet model.
    """
    # If the conf is a dataclass, convert it to a dictionary
    if dataclasses.is_dataclass(conf):
        conf = dataclasses.asdict(conf)
    # Instantiate the model
    if conf["architecture"] == "both":
        model = MultiONet
    elif conf["architecture"] == "branch":
        model = MultiONetB
    elif conf["architecture"] == "trunk":
        model = MultiONetT
    deeponet = model(
        conf["branch_input_size"],
        conf["hidden_size"],
        conf["branch_hidden_layers"],
        conf["trunk_input_size"],
        conf["hidden_size"],
        conf["trunk_hidden_layers"],
        conf["output_neurons"],
        conf["N_outputs"],
        device,
    )

    # Load the state dictionary
    if (
        "pretrained_model_path" not in conf or conf["pretrained_model_path"] is None
    ) and model_path is None:
        prev_train_loss = None
        prev_test_loss = None
    else:
        if model_path is None:
            model_path = conf["pretrained_model_path"]
        absolute_path = get_project_path(model_path)
        # Remove the .pth extension if it is present
        if absolute_path.endswith(".pth"):
            absolute_path = absolute_path[:-4]
        state_dict = torch.load(absolute_path + ".pth", map_location=device)
        deeponet.load_state_dict(state_dict)
        prev_losses = np.load(absolute_path + "_losses.npz")
        prev_train_loss = prev_losses["train_loss"]
        prev_test_loss = prev_losses["test_loss"]

    return deeponet, prev_train_loss, prev_test_loss


def test_deeponet(
    model: OperatorNetworkType,
    data_loader: DataLoader,
    device="cpu",
    criterion=nn.MSELoss(reduction="sum"),
    N_timesteps=101,
    timing=False,
    transpose=False,
    reshape=False,
) -> tuple:
    """
    Test a DeepONet model.

    :param model: A DeepONet model (as instantiated using the DeepONet class).
    :param data_loader: A DataLoader object.
    :param device: Device to use for testing.
    :param criterion: Loss function to use for testing.
    :param N_timesteps: Number of timesteps.
    :param timing: Whether to time the testing process.
    :param transpose: Whether to transpose the last two dimensions of the output arrays.
    :param reshape: Whether to reshape the output arrays.

    :return: Total loss and predictions.
    """
    device = torch.device(device)
    model.eval()
    model.to(device)

    # Calculate the total number of predictions to pre-allocate the buffer
    _, _, example_targets = next(iter(data_loader))
    dataset_size = len(data_loader.dataset)
    # Make sure the buffers have the correct shape for broadcasting
    # if len(example_targets.size()) == 1:
    #     targetsize = 1
    #     preds_buffer = np.empty(dataset_size)
    #     targets_buffer = np.empty(dataset_size)
    # else:
    #     targetsize = example_targets.size(1)
    #     preds_buffer = np.empty((dataset_size, targetsize))
    #     targets_buffer = np.empty((dataset_size, targetsize))

    targetsize = 1 if len(example_targets.size()) == 1 else example_targets.size(1)
    if targetsize == 1:
        preds_buffer = torch.empty(dataset_size, dtype=torch.float32)
        targets_buffer = torch.empty(dataset_size, dtype=torch.float32)
    else:
        preds_buffer = torch.empty(dataset_size, targetsize, dtype=torch.float32)
        targets_buffer = torch.empty(dataset_size, targetsize, dtype=torch.float32)

    buffer_index = 0
    total_loss = 0
    with torch.no_grad():
        start_time = time()
        for branch_inputs, trunk_inputs, targets in data_loader:
            if device != "cpu":
                branch_inputs = branch_inputs.to(device)
                trunk_inputs = trunk_inputs.to(device)
                targets = targets.to(device)
                model.to(device)
            outputs = model(branch_inputs, trunk_inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            # Store predictions in the buffer
            num_predictions = len(outputs)
            preds_buffer[buffer_index : buffer_index + num_predictions] = outputs
            targets_buffer[buffer_index : buffer_index + num_predictions] = targets
            buffer_index += num_predictions
        end_time = time()

    preds_buffer = preds_buffer.cpu().numpy()
    targets_buffer = targets_buffer.cpu().numpy()

    if timing:
        print(f"Testing time: {end_time - start_time:.2f} seconds")
        print(
            f"Average time per sample: {(end_time - start_time) * 1000 / dataset_size:.3f} ms"
        )

    # Calculate relative error
    total_loss /= dataset_size * targetsize

    if reshape:
        preds_buffer = preds_buffer.reshape(-1, N_timesteps, targetsize)
        targets_buffer = targets_buffer.reshape(-1, N_timesteps, targetsize)

    if transpose:
        preds_buffer = preds_buffer.transpose(0, 2, 1)
        targets_buffer = targets_buffer.transpose(0, 2, 1)

    return total_loss, preds_buffer, targets_buffer


@time_execution
def train_multionet_chemical(
    conf: type[dataclasses.dataclass],
    data_loader: DataLoader,
    test_loader: DataLoader = None,
) -> tuple:
    """Train a DeepONet model.
    The function instantiates a DeepONet model (with multiple outputs) and trains it using the provided DataLoader.
    Note that it assumes equal number of neurons in each hidden layer of the branch and trunk networks.

    Args:
        conf (type[dataclass]): A dataclass object containing the training configuration. It should have the following attributes:
            - 'masses' (List[float] | None): List of masses for the chemical species. If None, no mass conservation loss will be used.
            - 'branch_input_size' (int): Input size for the branch network.
            - 'trunk_input_size' (int): Input size for the trunk network.
            - 'hidden_size' (int): Number of hidden units in each layer.
            - 'branch_hidden_layers' (int): Number of hidden layers in the branch network.
            - 'trunk_hidden_layers' (int): Number of hidden layers in the trunk network.
            - 'output_size' (int): Number of neurons in the last layer.
            - 'N_outputs' (int): Number of outputs.
            - 'num_epochs' (int): Number of epochs to train for.
            - 'learning_rate' (float): Learning rate for the optimizer.
            - 'schedule' (bool): Whether to use a learning rate schedule.
            - 'N_sensors' (int): Number of sensor locations.
            - 'N_timesteps' (int): Number of timesteps.
            - 'architecture' (str): Architecture type, e.g., 'both', 'branch', or 'trunk'.
            - 'pretrained_model_path' (str | None): Path to a pretrained model. None if training from scratch.
            - 'device' (str): The device to use for training, e.g., 'cpu', 'cuda:0'.
            - 'use_streamlit' (bool): Whether to use Streamlit for live visualizations.
            - 'optuna_trial' (optuna.Trial | None): Optuna trial object for hyperparameter optimization. None if not using Optuna.
            - 'regularization_factor' (float): Regularization factor for the loss function.
            - 'massloss_factor' (float): Weight of the mass conservation loss component.
        data_loader (DataLoader): A DataLoader object containing the training data.
        test_loader (DataLoader): A DataLoader object containing the test data.

    :return: Trained DeepONet model and loss history.
    """
    device = torch.device(conf.device)
    print(f"Starting training on device {device}")

    deeponet, train_loss, test_loss = load_multionet(conf, device)

    criterion = setup_criterion(conf)

    optimizer, scheduler = setup_optimizer_and_scheduler(conf, deeponet)

    train_loss_hist, test_loss_hist = setup_losses(conf, train_loss, test_loss)
    output_hist = np.zeros((conf.num_epochs, 3, conf.N_sensors, conf.N_timesteps))

    progress_bar = tqdm(range(conf.num_epochs), desc="Training Progress")
    for epoch in progress_bar:
        train_loss_hist[epoch] = training_step(
            deeponet, data_loader, criterion, optimizer, device, conf.N_outputs
        )

        # if conf.optuna_trial is not None:
        #     conf.optuna_trial.report(train_loss_hist[epoch], epoch)
        #     if conf.optuna_trial.should_prune():
        #         raise optuna.TrialPruned()

        clr = optimizer.param_groups[0]["lr"]
        progress_bar.set_postfix({"loss": train_loss_hist[epoch], "lr": clr})
        scheduler.step()

        if test_loader is not None:
            test_loss_hist[epoch], outputs, targets = test_deeponet(
                deeponet,
                test_loader,
                device,
                criterion,
                conf.N_timesteps,
                reshape=True,
                transpose=True,
            )
            output_hist[epoch] = outputs[:3]

    if test_loader is None:
        test_loss_hist = None

    return deeponet, train_loss_hist, test_loss_hist
