import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from models import ModelWrapper
from data.lorenzo_data.lorenzo_data_utils import *
from params import DEVICE

latent_dim = 5
save_every = 5
n_epochs = 1000
learning_rate = 1e-3
batch_size = 256

t_range = torch.linspace(0, 1, 16).to(DEVICE)

losses_list = ["L2"] #["L2", "id", "deriv", "deriv2"]
loss_weights = torch.tensor(len(losses_list) * [1])

model = ModelWrapper(real_vars=10,
                     latent_vars=latent_dim,
                     ode_width = 128,
                     ode_hidden = 5,
                     tanh_reg = True,
                     coder_hidden = 4,
                     width_list = [32, 16, 8],
                     coder_activation=torch.nn.ReLU(),
                     loss_weights=loss_weights,
                     losses_list=losses_list).to(DEVICE)


optimizer = Adam(model.parameters(), lr=learning_rate)
scheduler = CosineAnnealingLR(optimizer, n_epochs, eta_min=1e-5)

dataset = LorenzoDatasetSmall()
dataset.load_data_from_original()
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in range(n_epochs + 1):
    for i, (x0, x_true) in enumerate(dataloader):
        optimizer.zero_grad()
        x_pred = model(x0, t_range)
        losses = model.total_loss(x_true, x_pred)
        loss = losses.sum()
        loss.backward()
        optimizer.step()
        with open("training2.log", "a") as f:
            f.write(f"{loss.item()}\n")

    torch.save(model.state_dict(), f"models/model_v2_{epoch}.pt")

# nohup .venv/bin/python3 training.py > training.out &