# Todos for the project

- [ ] **Write NeurIPS submission**

## Refactor
- [x] Remove the # model.n_timesteps = 100 lines in bench_fcts once new models are trained.
- [ ] Docstrings completion.
- [ ] Refactoring for more generality (remove chemistry specific code)
- [ ] **Find a catchy name** - CODESS Benchmark: Coupled ODE System Surrogates
- [ ] Think about the organisation of the model config files
- [ ] **Check the memory benchmarking**
- [ ] **Complete the README.md**
- [ ] **Make tutorial notebooks/examples**
- [ ] Rename NeuralODE to LatentNeuralODe
- [ ] Think about the error plots - where should absolute errors be used, where relative errors? Does it make sense to use relative errors for the chemical error plots?

## New Features
- [ ] **Javascript config maker**
- [x] **Make a table in the CLI at the end of the benchmark with the most important metrics**
- [x] **Make a csv/json file with the most important metrics**
- [ ] Inter-/extrapolation in the initial conditions (domain shift)
- [ ] Dataset Visualizations
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
- [ ] Contour plots to compare dynamics/UQ correlations
- [ ] Average prediction error over time + average gradient over time
- [x] **Add overall quantities to the plots (e.g. mean, std, etc.)**


