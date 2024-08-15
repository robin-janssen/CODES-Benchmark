# CODES Benchmark

## Benchmarking Coupled ODE Surrogates

This repo aims to provide a way of benchmarking surrogate models for coupled ODE systems, as can be found in the context of chemical reaction networks. 

Full documentation can be found at **add link here!**

## Motivation

There are many efforts to use machine learning models ("surrogates") to replace the costly numerics required involved in solving coupled ODEs. But for the end user, it is not obvious how to choose the right surrogate for a given task. Usually, the best choice depends on both the dataset and the target application.

Dataset specifics - how "complex" is the dataset?
- How many samples are there?
- Are the trajectories very dynamic or are the developments rather slow?
- How dense is the distribution of initial conditions?
- Is the data domain of interest well-covered by the domain of the training set?

Task requirements:
- What is the required accuracy?
- How important is inference time? Is the training time limited?
- Are there computational constraints (memory or processing power)?
- Is uncertainty estimation required (e.g. to replace uncertain predictions by numerics)?
- How much predictive flexibility is required? Do we need to interpolate or extrapolate across time?

Besides these practical considerations, one overarching question is always: Does the model only learn the data, or does it "understand" something about the underlying dynamics?

## Goals

This benchmark aims to aid in choosing the best surrogate model for the task at hand and additionally to shed some light on the above questions.

To achieve this, a selection of surrogate models are implemented in this repository. They can be trained on one of the included datasets or a custom dataset and then benchmarked on the corresponding test dataset. 

Some metrics included in the benchmark (but there is much more!):
- Absolute and relative error of the models.
- Inference time.
- Number of trainable parameters.
- Memory requirements (**WIP**).

Besides this, there are plenty of plots and visualisations providing insights into the models behaviour:

- Error distributions - per model, across time or per quantity.
- Insights into interpolation and extrapolation across time.
- Behaviour when training with sparse data or varying batch size.
- Predictions with uncertainty and predictive uncertainty across time.
- Correlations between the either predictive uncertainty or dynamics (gradients) of the data and the prediction error

Some prime **use-cases** of the benchmark are:
- Finding the best-performing surrogate on a dataset. Here, best-performing could mean high accuracy, low inference times or any other metric of interest (e.g. most accurate uncertainty estimates, ...).
- Comparing performance of a novel surrogate architecture against the implemented baseline models.
- Gaining insights into a dataset or comparing datasets using the built-in dataset insights. 

## Key Features

<details>
  <summary><b>Baseline Surrogates</b></summary>

The following surrogate models are currently implemented to be benchmarked:

- Fully Connected Neural Network: 
The vanilla neural network a.k.a. multilayer perceptron. 
- DeepONet: 
Two fully connected networks whose outputs are combined using a scalar product. In the current implementation, the surrogate comprises of only one DeepONet with multiple outputs (hence the name MultiONet).
- Latent NeuralODE: 
NeuralODE combined with an autoencoder that reduces the dimensionality of the dataset before solving the dynamics in the resulting latent space.
- Latent Polynomial: 
Uses an autoencoder similar to Latent NeuralODE, but fits a polynomial to the trajectories in the resulting latent space.

</details>

<details>
  <summary><b>Baseline Datasets</b></summary>

The following datasets are currently included in the benchmark:

</details>


<details>
  <summary><b>Uncertainty Quantification (UQ)</b></summary>

To give an uncertainty estimate that does not rely too much on the specifics of the surrogate architecture, we use DeepEnsemble for UQ. 

</details>

<details>
  <summary><b>Parallel Training</b></summary>

To gain insights into the surrogates behaviour, many models must be trained on varying subsets of the training data. This task is trivially parallelisable. In addition to utilising all specified devices, the benchmark features some nice progress bars to gain insights into the current status of the training. 

</details>


<details>
  <summary><b>Plots, Plots, Plots</b></summary>

While hard metrics are crucial to compare the surrogates, performance cannot always be broken down to a set of numbers. Running the benchmark creates many plots that serve to compare performance of surrogates or provide insights into the performance of each surrogate.

</details>

<details>
  <summary><b>Dataset Insights (WIP)</b></summary>

"Know your data" is one of the most important rules in machine learning. To aid in this, the benchmark provides plots and visualisations that should help to understand the dataset better.

</details>

<details>
  <summary><b>Tabular Benchmark Results</b></summary>

At the end of the benchmark, the most important metrics are displayed in a table, additionally, all metrics generated during the benchmark are provided as a csv file.

</details>

<details>
  <summary><b>Reproducibility</b></summary>

Randomness is an important part of machine learning and even required in the context of UQ with DeepEnsemble, but reproducibility is key in benchmarking enterprises. The benchmark uses a custom seed that can be set by the user to ensure full reproducibility.

</details>

<details>
  <summary><b>Custom Datasets and Own Models</b></summary>

To cover a wide variety of use-cases, the benchmark is designed such that adding own datasets and models is explicitly supported.

</details>

## Quickstart

## Usage

**Move this part to the docs page**

<details>
  <summary><i>Setup</i></summary>
</details>

<details>
  <summary><i>Running the Benchmark (Plots + Metrics)</i></summary>
</details>

<details>
  <summary><i>Benchmark Results</i></summary>
</details>


<details>
  <summary><i>Adding a new model</i></summary>

A new model should be implemented as a subclass to the base class AbstractSurrogateModel. This class in turn is a subclass of nn.Module, such that each model will be a nn.Module also. 

AbstractSurrogateModel mandates the implementation of five methods that are required either for training or for benchmarking. Please ensure that any implemented model adheres to the definition of these methods regarding the number of inputs and outputs as well as their data type and shape. This is import for the train.py and benchmark.py scripts to run. 

The methods are:

* forward(inputs):
  Implements one forward pass of the model. 

  Inputs:
  * inputs (tuple): Tuple of N torch.tensors as returned by the dataloader (i.e. inputs = next(iter(dataloader))). This means that inputs will contain, in addition to the initial conditions and the times, the targets. The targets can be discarded here, as they are not needed for the forward pass. This handling is for compatibility with the training and benchmarking scripts.

  Returns:
  * preds (torch.tensor): Predicted (chemical) quantities at the specified times, shape [batch_size, n_chemicals]

- prepare_data(data, timesteps, shuffle=True):

  This function should prepare the provided data for use with the predict function. This usually means creating a (torch) DataLoader from the given data. The shuffling is optional since we also want to use this to prepare test data, where the order of the predictions does not matter (it could be advantageous to not have the test data shuffled, e.g. for plotting).

  Inputs:
  * data (numpy.ndarray): Data to prepare, shape [n_samples, n_chemicals, n_timesteps].
  * timesteps (np.ndarray): Timesteps corresponding to the data, shape [n_timesteps].
  * shuffle (bool): Whether to shuffle the data before returning it. Default is True. Note: When using a DataLoader, the shuffle option should be passed on to the DataLoader rather than shuffling the numpy array and then creating the DataLoader.

  Returns:
  * dataloader (torch.utils.data.DataLoader): DataLoader with the prepared data. The batches in the DataLoader should be a triple (initial_conditions, times, targets), where initial_conditions and times are the inputs to the model and targets are the true values to compare the predictions to.

- fit(train_loader, test_loader, timesteps): 

  This is the training implementation for the model. It receives three inputs. Two dataloaders (train_loader and test_loader) and the timesteps in the form of a numpy array.

  Inputs:
  * train_loader (torch.utils.data.DataLoader): DataLoader with the training data.
  * test_loader (torch.utils.data.DataLoader): DataLoader with the test data.
  * timesteps (np.ndarray): Timesteps corresponding to the data, shape [n_timesteps].

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

  This method is used to make predictions for the provided dataloader. The predictions are compared to the true values using the provided criterion.

  Note: In the context of the benchmark, the criterion used is torch.nn.MSELoss(reduction="sum"). This means that across a batch, the loss is the sum of the squared differences between the predictions and the true values. Since we are interested in the mean squared error per prediction, the sum of these losses should be divided by the number of predictions, which is the product of the batch size, timesteps and number of chemicals. This quantity is returned as the total_loss.

  Inputs:
  * dataloader (torch.utils.data.DataLoader): DataLoader with the data to make predictions on.
  * criterion (torch.nn.Module): Loss function to use for comparing the predictions to the true values.
  * timesteps (np.ndarray): Timesteps corresponding to the data, shape [n_timesteps].

  Returns:
  * total_loss (float): Total loss of the predictions.
  * preds (torch.tensor): Predictions made by the model, shape [n_samples, n_chemicals, n_timesteps].
  * targets (torch.tensor): True values to compare the predictions to, shape [n_samples, n_chemicals, n_timesteps].


</details>

<details>
  <summary><i>Adding a new dataset</i></summary>


It is easy to add a new dataset to the benchmark. To do so, you can use the function create_hdf5_dataset (train_data, test_data, dataset_name, data_dir, timesteps). An example of its usage is given in the script make_new_dataset.py.

The function takes five inputs:

* train_data (numpy.ndarray): Training data to save, shape [n_samples, n_chemicals, n_timesteps].
* test_data (numpy.ndarray): Test data to save, shape [n_samples, n_chemicals, n_timesteps].
* dataset_name (str): Name of the dataset to save.
* data_dir (str): Directory to save the dataset in.
* timesteps (np.ndarray): Timesteps corresponding to the data, shape [n_timesteps].

It is important that the train and test data have the correct shape, i.e. [n_samples, n_chemicals, n_timesteps]. 
The default data_dir is data, which is a subdirectory of the root directory of the repository. The timesteps are an optional input, if they are not provided, the check_and_load_data function will automatically create a numpy array with the timesteps from 0 to n_timesteps - 1.

The data is stored as provided in the numpy arrays. Please ensure that the data is clean and has a reasonable range. Ideally, it should be normalized. The data is stored in the path data_dir/dataset_name/data.hdf5.  

</details>

<details>
  <summary><i>Repo Structure</i></summary>
</details>

<details>
  <summary><i>Code Documentation</i></summary>
</details>