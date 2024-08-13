import torch


class ChemDataset(torch.utils.data.Dataset):

    def __init__(self, raw_data, timesteps, device):
        self.data = torch.tensor(raw_data, dtype=torch.float64)
        # self.xmin = self.data.min() if xmin is None else xmin
        # self.xmax = self.data.max() if xmax is None else xmax
        # self.data = 2 * (self.data - self.xmin) / (self.xmax - self.xmin) - 1
        self.length = self.data.shape[0]
        if not self.data.dtype == torch.float64:
            self.data = torch.tensor(self.data, dtype=torch.float64)
        self.data = self.data.to(device)
        self.timesteps = timesteps

    def __getitem__(self, index):
        return self.data[index, :, :], self.timesteps

    def __getitems__(self, index_list: list[int]):
        return self.data[index_list, :, :], self.timesteps

    def __len__(self):
        return self.length
