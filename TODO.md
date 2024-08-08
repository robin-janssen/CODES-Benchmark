# Todos for the project

- [ ] **Write NeurIPS submission**

## Refactor
- [ ] Remove the model.n_timesteps = 100 lines in bench_fcts once new models are trained.
- [ ] Docstrings completion.
- [ ] Refactoring for more generality (remove chemistry specific code)
- [ ] **Find a catchy name**
- [ ] Think about the organisation of the model config files
- [ ] **Check the memory benchmarking**
- [ ] **Complete the README.md**
- [ ] **Make tutorial notebooks/examples**
- [ ] Rename NeuralODE to LatentNeuralODe

## New Features
- [ ] **Javascript config maker**
- [ ] **Make a table in the CLI at the end of the benchmark with the most important metrics**
- [ ] **Make a csv/json file with the most important metrics**
- [ ] Inter-/extrapolation in the initial conditions (domain shift)
- [ ] Dataset Visualizations
- [ ] Surrogate performance on Baseline ODE systems (Lorenz, Van der Pol, etc.) -> implement dynamic datasets
- [ ] Data Storage (heibox?)
- [ ] Datasets by Lorenzo and Simon Glover
- [ ] Optuna tuning script
- [ ] Determine optimal model parameters for the baseline models per dataset

## Potential Models
- [ ] Multiple DeepONets
- [ ] NeuralODE without autoencoder
- [ ] LSTM
- [ ] SINDy 
- [ ] Chemulator

## Plots
- [ ] Heatmaps comparative plot
- [ ] Layout of chemical error distribution plot similar to example UQ plots
- [ ] Contour plots to compare dynamics/UQ correlations
- [ ] **Add overall quantities to the plots (e.g. mean, std, etc.)**

