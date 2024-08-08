from surrogates import NeuralODE
from surrogates.NeuralODE.neural_ode_config import NeuralODEConfigOSU as Config
from data import check_and_load_data

DEVICE = "cuda:4"
EPOCHS = 5

model = NeuralODE(DEVICE, 29, 100)

train_data, test_data, _, timesteps, _, data_params = check_and_load_data(
        dataset_name="osu2008",
        verbose=False,
        log=False,
        normalisation_mode="minmax",
    )

train_loader, test_loader, _ = model.prepare_data(
        dataset_train=train_data,
        dataset_test=test_data,
        dataset_val=None,
        timesteps=timesteps,
        batch_size=128,
        shuffle=True,
    )

model.fit(train_loader, test_loader, timesteps, EPOCHS)
print("finished")
