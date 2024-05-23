from typing import TypeVar

import torch
import torch.nn as nn


class OperatorNetwork(nn.Module):
    def __init__(self):
        super(OperatorNetwork, self).__init__()

    def post_init_check(self):
        if not hasattr(self, "branch_net") or not hasattr(self, "trunk_net"):
            raise NotImplementedError(
                "Child classes must initialize a branch_net and trunk_net."
            )
        if not hasattr(self, "forward") or not callable(self.forward):
            raise NotImplementedError("Child classes must implement a forward method.")

    def forward(self, branch_input, trunk_input):
        # Define a generic forward pass or raise an error to enforce child class implementation
        raise NotImplementedError("Forward method must be implemented by subclasses.")

    @staticmethod
    def _calculate_split_sizes(total_neurons, num_splits):
        """Helper function to calculate split sizes for even distribution"""
        base_size = total_neurons // num_splits
        remainder = total_neurons % num_splits
        return [
            base_size + 1 if i < remainder else base_size for i in range(num_splits)
        ]


OperatorNetworkType = TypeVar("OperatorNetworkType", bound=OperatorNetwork)


class BranchNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers):
        super(BranchNet, self).__init__()
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        layers.append(nn.Linear(hidden_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TrunkNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers):
        super(TrunkNet, self).__init__()
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        layers.append(nn.Linear(hidden_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class DeepONet(OperatorNetwork):
    def __init__(
        self,
        inputs_b,
        hidden_b,
        layers_b,
        inputs_t,
        hidden_t,
        layers_t,
        outputs,
        device,
    ):
        super(DeepONet, self).__init__()
        self.branch_net = BranchNet(inputs_b, hidden_b, outputs, layers_b).to(device)
        self.trunk_net = TrunkNet(inputs_t, hidden_t, outputs, layers_t).to(device)

    def forward(self, branch_input, trunk_input):
        branch_output = self.branch_net(branch_input)
        trunk_output = self.trunk_net(trunk_input)
        return torch.sum(branch_output * trunk_output, dim=1)


class MultiONet(OperatorNetwork):
    def __init__(
        self,
        inputs_b,
        hidden_b,
        layers_b,
        inputs_t,
        hidden_t,
        layers_t,
        outputs,
        N,
        device,
    ):
        super(MultiONet, self).__init__()
        self.N = N  # Number of outputs
        self.outputs = outputs  # Number of neurons in the last layer
        self.branch_net = BranchNet(inputs_b, hidden_b, outputs, layers_b).to(device)
        self.trunk_net = TrunkNet(inputs_t, hidden_t, outputs, layers_t).to(device)

    def forward(self, branch_input, trunk_input):
        branch_output = self.branch_net(branch_input)
        trunk_output = self.trunk_net(trunk_input)

        # Splitting the outputs for multiple output values
        split_sizes = self._calculate_split_sizes(self.outputs, self.N)
        branch_splits = torch.split(branch_output, split_sizes, dim=1)
        trunk_splits = torch.split(trunk_output, split_sizes, dim=1)

        result = []
        for b_split, t_split in zip(branch_splits, trunk_splits):
            result.append(torch.sum(b_split * t_split, dim=1, keepdim=True))

        return torch.cat(result, dim=1)


class MultiONetB(OperatorNetwork):
    def __init__(
        self,
        inputs_b,
        hidden_b,
        layers_b,
        inputs_t,
        hidden_t,
        layers_t,
        outputs,
        N,
        device,
    ):
        super(MultiONetB, self).__init__()
        self.N = N  # Number of outputs
        self.outputs = outputs  # Number of neurons in the last layer
        trunk_outputs = outputs // N

        # Initialize branch and trunk networks
        self.branch_net = BranchNet(inputs_b, hidden_b, outputs, layers_b).to(device)
        self.trunk_net = TrunkNet(inputs_t, hidden_t, trunk_outputs, layers_t).to(
            device
        )

    def forward(self, branch_input, trunk_input):
        branch_output = self.branch_net(branch_input)
        trunk_output = self.trunk_net(trunk_input)

        # Splitting the outputs for multiple output values
        split_sizes = self._calculate_split_sizes(self.outputs, self.N)
        branch_splits = torch.split(branch_output, split_sizes, dim=1)

        result = []
        for b_split in branch_splits:
            result.append(torch.sum(b_split * trunk_output, dim=1, keepdim=True))

        return torch.cat(result, dim=1)


class MultiONetT(OperatorNetwork):
    def __init__(
        self,
        inputs_b,
        hidden_b,
        layers_b,
        inputs_t,
        hidden_t,
        layers_t,
        outputs,
        N,
        device,
    ):
        super(MultiONetT, self).__init__()
        self.N = N  # Number of outputs
        self.outputs = outputs  # Number of neurons in the last layer
        branch_outputs = outputs // N

        # Initialize branch and trunk networks
        self.branch_net = BranchNet(inputs_b, hidden_b, branch_outputs, layers_b).to(
            device
        )
        self.trunk_net = TrunkNet(inputs_t, hidden_t, outputs, layers_t).to(device)

    def forward(self, branch_input, trunk_input):
        branch_output = self.branch_net(branch_input)
        trunk_output = self.trunk_net(trunk_input)

        # Splitting the outputs for multiple output values
        split_sizes = self._calculate_split_sizes(self.outputs, self.N)
        trunk_splits = torch.split(trunk_output, split_sizes, dim=1)

        result = []
        for t_split in trunk_splits:
            result.append(torch.sum(branch_output * t_split, dim=1, keepdim=True))

        return torch.cat(result, dim=1)


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.zeros_(m.bias.data)
