# import functools
# import time
# import os
import torch.utils

# import yaml
# import dataclasses

# import numpy as np
import torch
import torch.nn as nn

# import torch.optim as optim

# from .deeponet import OperatorNetworkType
# from .utils import create_date_based_directory


def mass_conservation_loss(
    masses: list,
    criterion=nn.MSELoss(reduction="sum"),
    weights: tuple = (1, 1),
    device: torch.device = torch.device("cpu"),
):
    """
    Replaces the standard MSE loss with a sum of the standard MSE loss and a mass conservation loss.

    :param masses: A list of masses for the chemical species.
    :param criterion: The loss function to use for the standard loss.
    :param weights: A 2-tuple of weights for the standard loss and the mass conservation loss.
    :param device: The device to use for the loss function.

    :return: A new loss function that includes the mass conservation loss.
    """
    masses = torch.tensor(masses, dtype=torch.float32, device=device)

    def loss(outputs: torch.tensor, targets: torch.tensor) -> torch.tensor:
        """
        Loss function that includes the mass conservation loss.

        :param outputs: The predicted values.
        :param targets: The ground truth values.
        """
        standard_loss = criterion(outputs, targets)

        # Calculate the weighted sum of each chemical quantity for predicted and ground truth,
        # resulting in the total predicted mass and ground truth mass for each sample in the batch
        predicted_mass = torch.sum(outputs * masses, dim=1)
        true_mass = torch.sum(targets * masses, dim=1)

        # Calculate the mass conservation loss as the MSE of the predicted mass vs. true mass
        mass_loss = torch.abs(predicted_mass - true_mass).sum()
        # Sum up the standard MSE loss and the mass conservation loss
        total_loss = weights[0] * standard_loss + weights[1] * mass_loss

        # print(f"Standard loss: {standard_loss.item()}, Mass loss: {mass_loss.item()}")

        return total_loss

    return loss


# def poly_eval_torch(p, x):
#     """
#     Evaluate a polynomial at given points in PyTorch.

#     :param p: A tensor of shape [batch_size, n_coeffs] containing the coefficients of the polynomial.
#     :param x: A tensor of shape [n_points] containing the x-values at which to evaluate the polynomial.
#     :return: A tensor of shape [batch_size, n_points] with the evaluation of the polynomial at the x-values.
#     """
#     n = p.shape[1]  # Number of coefficients
#     x = x.unsqueeze(0).repeat(p.shape[0], 1)  # Shape [batch_size, n_points]
#     powers = torch.arange(
#         n - 1, -1, -1, device=p.device
#     )  # Exponents for each coefficient
#     x_powers = x.unsqueeze(-1).pow(powers)  # Shape [batch_size, n_points, n_coeffs]
#     return torch.sum(p.unsqueeze(1) * x_powers, dim=-1)  # Polynomial evaluation


# def time_execution(func):
#     """
#     Decorator to time the execution of a function and store the duration
#     as an attribute of the function.
#     """

#     @functools.wraps(func)
#     def wrapper(*args, **kwargs):
#         start_time = time.time()
#         result = func(*args, **kwargs)
#         end_time = time.time()
#         wrapper.duration = end_time - start_time
#         print(f"{func.__name__} executed in {wrapper.duration:.2f} seconds.")
#         return result

#     wrapper.duration = None
#     return wrapper


# def save_model_2(
#     model: OperatorNetworkType,
#     model_name,
#     hyperparameters,
#     subfolder="models",
#     train_loss: np.ndarray | None = None,
#     test_loss: np.ndarray | None = None,
# ):
#     """
#     Save the trained model and hyperparameters.

#     :param model: The trained model.
#     :param hyperparameters: Dictionary containing hyperparameters.
#     :param base_dir: Base directory for saving the model.
#     """
#     # Create a directory based on the current date
#     base_dir = os.path.dirname(os.path.realpath(__file__))
#     model_dir = create_date_based_directory(base_dir, subfolder)

#     # Save the model state dict
#     model_path = os.path.join(model_dir, f"{model_name}.pth")
#     torch.save(model.state_dict(), model_path)

#     # Save hyperparameters as a YAML file
#     hyperparameters_path = os.path.join(model_dir, f"{model_name}.yaml")
#     with open(hyperparameters_path, "w") as file:
#         yaml.dump(hyperparameters, file)

#     if train_loss is not None and test_loss is not None:
#         # Save the losses as a numpy file
#         losses_path = os.path.join(model_dir, f"{model_name}_losses.npz")
#         np.savez(losses_path, train_loss=train_loss, test_loss=test_loss)

#     print(f"Model, losses and hyperparameters saved to {model_dir}")


# def save_model(
#     model: OperatorNetworkType,
#     model_name: str,
#     config: type[dataclasses.dataclass],
#     subfolder: str = "models",
#     train_loss: np.ndarray | None = None,
#     test_loss: np.ndarray | None = None,
#     training_duration: float | None = None,
# ) -> None:
#     """
#     Save the trained model and hyperparameters.

#     Args:
#         model: The trained model.
#         model_name: The name of the model.
#         config: Dictionary containing hyperparameters.
#         subfolder: The subfolder to save the model in.
#         train_loss: The training loss history.
#         test_loss: The testing loss history.
#         training_duration: The duration of the training.
#     """
#     # Create a directory based on the current date
#     base_dir = os.getcwd()
#     model_dir = create_date_based_directory(base_dir, subfolder)

#     # Save the model state dict
#     model_path = os.path.join(model_dir, f"{model_name}.pth")
#     torch.save(model.state_dict(), model_path)

#     # Create the hyperparameters dictionary from the config dataclass
#     hyperparameters = dataclasses.asdict(config)

#     # Append the train time to the hyperparameters
#     hyperparameters["train_duration"] = training_duration

#     # Save hyperparameters as a YAML file
#     hyperparameters_path = os.path.join(model_dir, f"{model_name}.yaml")
#     with open(hyperparameters_path, "w") as file:
#         yaml.dump(hyperparameters, file)

#     if train_loss is not None and test_loss is not None:
#         # Save the losses as a numpy file
#         losses_path = os.path.join(model_dir, f"{model_name}_losses.npz")
#         np.savez(losses_path, train_loss=train_loss, test_loss=test_loss)

#     print(f"Model, losses and hyperparameters saved to {model_dir}")


# def setup_optimizer_and_scheduler(
#     conf: type[dataclasses.dataclass], deeponet: OperatorNetworkType
# ) -> tuple:
#     """
#     Utility function to set up the optimizer and scheduler for training.

#     Args:
#         conf (dataclasses.dataclass): The configuration dataclass.
#         deeponet (OperatorNetworkType): The model to train.

#     Returns:
#         tuple (torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler): The optimizer and scheduler.
#     """
#     optimizer = optim.Adam(
#         deeponet.parameters(),
#         lr=conf.learning_rate,
#         weight_decay=conf.regularization_factor,
#     )
#     if conf.schedule:
#         scheduler = optim.lr_scheduler.LinearLR(
#             optimizer, start_factor=1, end_factor=0.3, total_iters=conf.num_epochs
#         )
#     else:
#         scheduler = optim.lr_scheduler.LinearLR(
#             optimizer, start_factor=1, end_factor=1, total_iters=conf.num_epochs
#         )
#     return optimizer, scheduler


# def setup_criterion(conf: type[dataclasses.dataclass]) -> callable:
#     """
#     Utility function to set up the loss function for training.

#     Args:
#         conf (dataclasses.dataclass): The configuration dataclass.

#     Returns:
#         callable: The loss function.
#     """
#     crit = nn.MSELoss(reduction="sum")
#     if hasattr(conf, "masses") and conf.masses is not None:
#         weights = (1.0, conf.massloss_factor)
#         crit = mass_conservation_loss(conf.masses, crit, weights, conf.device)
#     return crit


# def setup_losses(
#     conf: type[dataclasses.dataclass],
#     prev_train_loss: np.ndarray,
#     prev_test_loss: np.ndarray,
# ) -> tuple:
#     """
#     Utility function to set up the loss history arrays for training.

#     Args:
#         conf (dataclasses.dataclass): The configuration dataclass.

#     Returns:
#         tuple: The training and testing loss history arrays (both np.ndarrays).
#     """
#     if conf.pretrained_model_path is None:
#         train_loss_hist = np.zeros(conf.num_epochs)
#         test_loss_hist = np.zeros(conf.num_epochs)
#     else:
#         train_loss_hist = np.concatenate((prev_train_loss, np.zeros(conf.num_epochs)))
#         test_loss_hist = np.concatenate((prev_test_loss, np.zeros(conf.num_epochs)))

#     return train_loss_hist, test_loss_hist


# def training_step(model, data_loader, criterion, optimizer, device, N_outputs=1):
#     """
#     Perform a single training step on the model.

#     Args:
#         model (OperatorNetworkType): The model to train.
#         data_loader (torch.utils.data.DataLoader): The data loader for the training data.
#         criterion (torch.nn.Module): The loss function.
#         optimizer (torch.optim.Optimizer): The optimizer.
#         device (torch.device): The device to use for training.
#         N_outputs (int): The number of outputs of the model.

#     Returns:
#         float: The total loss for the training step.
#     """
#     model.train()
#     total_loss = 0
#     dataset_size = len(data_loader.dataset)
#     for branch_inputs, trunk_inputs, targets in data_loader:
#         branch_inputs, trunk_inputs, targets = (
#             branch_inputs.to(device),
#             trunk_inputs.to(device),
#             targets.to(device),
#         )

#         optimizer.zero_grad()
#         outputs = model(branch_inputs, trunk_inputs)
#         loss = criterion(outputs, targets)
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()
#     total_loss /= dataset_size * N_outputs
#     return total_loss


# def inference_timing(
#     deeponet: OperatorNetworkType, data_loader: torch.utils.data.DataLoader, device: str
# ) -> None:
#     """
#     Measure the inference time of the model.

#     Args:
#         deeponet (OperatorNetworkType): The model to test.
#         data_loader (torch.utils.data.DataLoader): The data loader for the test data.
#         device (str): The device to use for testing.
#     """
#     deeponet.eval()
#     with torch.no_grad():
#         inference_times = []
#         for branch_inputs, trunk_inputs, _ in data_loader:
#             branch_inputs, trunk_inputs = branch_inputs.to(device), trunk_inputs.to(
#                 device
#             )
#             deeponet = deeponet.to(device)
#             start_time = time.time()
#             _ = deeponet(branch_inputs, trunk_inputs)
#             end_time = time.time()
#             inference_times.append(end_time - start_time)

#     avg_inference_time = np.mean(inference_times[1:])
#     std_inference_time = np.std(inference_times[1:])
#     print(
#         f"Average inference time: {avg_inference_time:.4f} ± {std_inference_time:.4f} s on {device}"
#     )
