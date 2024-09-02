import torch


class ChemDataset(torch.utils.data.Dataset):
    """
    Dataset class for the latent neural ODE model. The data is
    a 3D tensor with dimensions (batch, timesteps, species). The
    dataset also returns the timesteps for the data, as they are
    requred for the integration.
    """

    def __init__(self, raw_data, timesteps, device):
        self.data = torch.tensor(raw_data, dtype=torch.float64)
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
