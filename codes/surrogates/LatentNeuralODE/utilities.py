import torch


class ChemDataset(torch.utils.data.Dataset):
    """
    Dataset class for the latent neural ODE model.
    Returns each sample along with its timesteps and (optionally) fixed parameters.
    """

    def __init__(self, raw_data, timesteps, device, parameters):
        if not isinstance(raw_data, torch.Tensor):
            raw_data = torch.tensor(raw_data, dtype=torch.float64)
        self.data = raw_data.to(device)
        self.length = self.data.shape[0]
        if not isinstance(timesteps, torch.Tensor):
            timesteps = torch.tensor(timesteps, dtype=torch.float64)
        self.timesteps = timesteps.to(device)
        if parameters is not None:
            if not isinstance(parameters, torch.Tensor):
                parameters = torch.tensor(parameters, dtype=torch.float64)
            self.parameters = parameters.to(device)
        else:
            self.parameters = None

    def __getitem__(self, index):
        if self.parameters is not None:
            return self.data[index, :, :], self.timesteps, self.parameters[index]
        else:
            return self.data[index, :, :], self.timesteps

    def __len__(self):
        return self.length
