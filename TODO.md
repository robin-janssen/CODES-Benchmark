# Todos for the project

- [ ] **Write NeurIPS submission**
- [ ] Read into DataHosting on HeiDATA
- [ ] Choose a code licence for the code - MIT?
- [ ] Copy over actions and other settings from the SSC Cookiecutter Template

## Refactor
- [x] Remove the # model.n_timesteps = 100 lines in bench_fcts once new models are trained.
- [x] **Find a catchy name** - CODES Benchmark: Coupled ODE System Surrogates
- [x] Think about the organisation of the model config files
- [x] **Complete the README.md**
- [x] Rename NeuralODE to LatentNeuralODE
- [ ] **Check the memory benchmarking**
- [ ] **Check the time benchmarking** How should we measure the time?
- [ ] **Save Plots as SVG for immaculate quality**
- [ ] Latex-font labels in the plots?
- [ ] Find a way to save the optimal batch size and recommended number of epochs for each model.
- [ ] Rename batch_scaling to batch.
- [ ] Add user prompt on whether to use task list or overwrite it.
- [ ] Check whether the training is indeed deterministic.
- [ ] Docstrings completion.
- [ ] Make a clean requirements.txt
- [ ] Refactoring for more generality (remove chemistry specific code)
- [ ] **Make tutorial notebooks/examples**
- [ ] Think about the error plots - where should absolute errors be used, where relative errors? Does it make sense to use relative errors for the chemical error plots?
- [ ] Add testing
- [ ] **Clean up the python environment and do a fresh test install**

## New Features
- [x] **Javascript config maker**
- [x] **Make a table in the CLI at the end of the benchmark with the most important metrics**
- [x] **Make a csv/json file with the most important metrics**
- [ ] Calculate and add error quantities per surrogate to the individal (and comparative) outputs.
- [ ] Continue training of the existing models in case further convergence is needed
- [ ] Accumulation of error when predicting quantities iteratively
- [ ] Inter-/extrapolation in the initial conditions (domain shift)
- [ ] Dataset Visualizations - also include how the distribution changes over time (for iterative preds)
- [ ] Surrogate performance on Baseline ODE systems (Lorenz, Van der Pol, etc.) -> implement dynamic datasets
- [ ] Data Storage (heibox?)
- [ ] Datasets by Lorenzo and Simon Glover
- [ ] Optuna tuning script
- [ ] Determine optimal model parameters for the baseline models per dataset
- [ ] Integrate Torch Compile for potentially better performance

## Potential Models
- [ ] Multiple DeepONets
- [ ] NeuralODE without autoencoder
- [ ] LSTM
- [ ] SINDy 
- [ ] Chemulator

## Plots
- [x] Heatmaps comparative plot
- [x] Layout of chemical error distribution plot similar to example UQ plots
- [x] **Add overall quantities to the plots (e.g. mean, std, etc.)**
- [ ] Contour plots to compare dynamics/UQ correlations
- [ ] Average prediction error over time + average gradient over time
- [ ] Scatter plot of absolute/relative error vs. inference time comparing surrogates.


