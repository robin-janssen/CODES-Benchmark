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

- forward(initial_conditions, times):
Implements one forward pass of the model. 
Inputs:

- prepare_data(data, timesteps, shuffle=True):
This function should prepare the provided data for use with the predict function. This usually means creating a (torch) DataLoader from the given data. The shuffling is optional since we also want to use this to prepare test data, where the order of the predictions does not matter (it could be advantageous to not have the test data shuffled, e.g. for plotting)
- fit(train_loader, test_loader, timesteps): 
This is the training implementation for the model. It receives three inputs. Two dataloaders (train_loader and test_loader) and the timesteps in the form of a numpy array.
- save(model_name, unique_id, dataset_name):
This method is used to save the model, usually the statedict. The convention for loading the model 
- predict(dataloader, criterion, timesteps):
This method is used to make predictions for the provided dataloader. 


</details>

<details>
  <summary><i>Adding a new dataset</i></summary>
</details>