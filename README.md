# CODEBench
## Chemical ODE surrogate benchmark

This repo aims to provide a way of benchmarking surrogate models for chemical reaction networks, i.e. coupled ODEs.

## Goals

## Structure

## Usage



<details>
  <summary><i>Adding a new model</i></summary>

A new model should be implemented as a subclass to the base class AbstractSurrogateModel. This class in turn is a subclass of nn.Module, such that each model will be an nn.Module also. 

AbstractSurrogateModel mandates the implementation of four methods that are required either for training or for benchmarking. The methods are:

* forward(initial_conditions, times):
  Implements one forward pass of the model. 

  Inputs:
  * initial_conditions (torch.tensor): (Chemical) quantities at t=0, shape [batch_size, N_chemicals].
  * times (torch.tensor): Times at which to obtain the predictions, shape [batch_size, 1]

  Returns:
  * preds (torch.tensor): Predicted (chemical) quantities at the specified times, shape [batch_size, N_chemicals]

- prepare_data(data, timesteps, shuffle=True):

  This function should prepare the provided data for use with the predict function. This usually means creating a (torch) DataLoader from the given data. The shuffling is optional since we also want to use this to prepare test data, where the order of the predictions does not matter (it could be advantageous to not have the test data shuffled, e.g. for plotting).

  Inputs:
  * data (numpy.ndarray): Data to prepare, shape [N_samples, N_chemicals, N_timesteps].
  * timesteps (np.ndarray): Timesteps corresponding to the data, shape [N_timesteps].
  * shuffle (bool): Whether to shuffle the data before returning it. Default is True. Note: When using a DataLoader, the shuffle option should be passed on to the DataLoader rather than shuffling the numpy array and then creating the DataLoader.

  Returns:
  * dataloader (torch.utils.data.DataLoader): DataLoader with the prepared data. The batches in the DataLoader should be a triple (initial_conditions, times, targets), where initial_conditions and times are the inputs to the model and targets are the true values to compare the predictions to.

- fit(train_loader, test_loader, timesteps): 

  This is the training implementation for the model. It receives three inputs. Two dataloaders (train_loader and test_loader) and the timesteps in the form of a numpy array.

  Inputs:
  * train_loader (torch.utils.data.DataLoader): DataLoader with the training data.
  * test_loader (torch.utils.data.DataLoader): DataLoader with the test data.
  * timesteps (np.ndarray): Timesteps corresponding to the data, shape [N_timesteps].

  Returns:
  * None (the model is trained in place).
- save(model_name, training_id, dataset_name):
This method is used to save the model. In addition to the model parameters (the statedict), the train and test loss trajectories should be saved as a .npz file and the model configuration should be saved as a .yaml file.

  The convention to save models is to use the following format: trained/training_id/surrogate_name/model_name.pth, e.g. trained/training1/DeepONet/deeponet_sparse_2.pth. 

  Inputs:
  * model_name (str): Name of the model. Should not include the path or the file extension (e.g. deeponet_sparse_2)
  * training_id (str): Identifier for the training run (e.g. training1)
  * dataset_name (str): Name of the dataset that the model was trained on (e.g. Osu2008). This can be used to add the dataset name to the model config for later reference.

  Returns:
  * None (the model is saved to disk).

- predict(dataloader, criterion, timesteps):
This method is used to make predictions for the provided dataloader. 


</details>

<details>
  <summary><i>Adding a new dataset</i></summary>
</details>