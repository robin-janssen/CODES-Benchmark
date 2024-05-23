import torch
import numpy as np

class LorenzoDatasetSmall(torch.utils.data.Dataset):

    def __init__(self,
                 filepath = 'data/lorenzo_data/lorenzo_data_small.npy',
                 dtype = torch.float32,
                 device = 'cpu') -> None:
        self.device = device
        data = np.load(filepath)
        data[data == 0] = 1e-10
        data = np.log10(data)
        _min = data.min()
        _max = data.max()
        data = 2 * (data - _min) / (_max - _min) - 1
        self.data = torch.tensor(data, dtype=dtype).to(device)

    def __len__(self) -> int:
        return self.data.shape[0]
    
    def __getitem__(self, idx: int):
        return self.data[idx, 0, :], self.data[idx]
    
    def __getitems__(self, index_list):
        # performanter als einzelne items zu holen, wird automatisch von pytorch verwendet
        return self.data[torch.tensor(index_list, dtype=torch.int32)]