import torch


class ChemDataset(torch.utils.data.Dataset):
    """
    Dataset class for the latent neural ODE model. The data is
    a 3D tensor with dimensions (batch, timesteps, species). The
    dataset also returns the timesteps for the data, as they are
    requred for the integration.
    """

    def __init__(self, raw_data, timesteps, device):
        if not isinstance(raw_data, torch.Tensor):
            raw_data = torch.tensor(raw_data, dtype=torch.float64)
        else:
            raw_data = raw_data.to(torch.float64)
        self.data = raw_data
        self.length = self.data.shape[0]
        self.data = self.data.to(device)
        if not isinstance(timesteps, torch.Tensor):
            timesteps = torch.tensor(timesteps)
        self.timesteps = timesteps.to(device)

    def __getitem__(self, index):
        return self.data[index, :, :], self.timesteps

    def __getitems__(self, index_list: list[int]):
        return self.data[index_list, :, :], self.timesteps

    def __len__(self):
        return self.length
