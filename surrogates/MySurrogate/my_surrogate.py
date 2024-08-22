import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):

	def __init__(self, abundances, device):
		# abundances with shape (n_samples, n_timesteps, n_species)
		self.abundances = torch.tensor(abundances).to(device)
		self.length = self.abundances.shape[0]

	def __getitem__(self, index):
		return self.abundances[index, :, :]

	def __getitems__(self, index_list):	# for better batch performance, optional
		return self.abundances[index_list, :, :]

	def __len__(self):
		return self.length


