# Todos for the project

- [ ] **Write NeurIPS submission**
- [x] Read into DataHosting on HeiDATA -> use Zenodo for now
- [ ] Choose a code licence for the code - MIT?
- [ ] Reach out to Lorenzo regarding the dataset
- [ ] Copy over actions and other settings from the SSC Cookiecutter Template

## Refactor
- [x] Remove the # model.n_timesteps = 100 lines in bench_fcts once new models are trained.
- [x] **Find a catchy name** - CODES Benchmark: Coupled ODE System Surrogates
- [x] Think about the organisation of the model config files
- [x] **Complete the README.md**
- [x] Rename NeuralODE to LatentNeuralODE
- [x] **Check the time benchmarking**
- [ ] **Check the memory benchmarking**
- [ ] **Save Plots as SVG for immaculate quality**
- [ ] Rename batch_scaling to batch.
- [ ] **Check whether the training is indeed deterministic.**
- [ ] **Docstrings completion.**
- [ ] **Make a clean requirements.txt, clean up the python environment and do a fresh test install**
- [ ] **Make tutorial notebooks/examples**
- [ ] **Check/update the config maker**
- [ ] Think about the error plots - where should absolute errors be used, where relative errors? Does it make sense to use relative errors for the chemical error plots?
- [ ] **Add testing (for workshop)**

## New Features
- [x] **Javascript config maker**
- [x] **Make a table in the CLI at the end of the benchmark with the most important metrics**
- [x] **Make a csv file with the most important metrics**
- [x] **Make a json file with the most important metrics**
- [ ] **Add and document Lorenzo's Dataset**
- [x] Calculate and add error quantities per surrogate to the individal (and comparative) outputs.
- [ ] Store output in a .txt file
- [ ] Refactoring for more generality (remove chemistry specific code)
- [ ] Add user prompt on whether to use task list or overwrite it.
- [ ] Continue training of the existing models in case further convergence is needed
- [ ] Accumulation of error when predicting quantities iteratively
- [ ] Inter-/extrapolation in the initial conditions (domain shift)
- [ ] Dataset Visualizations - also include how the distribution changes over time (for iterative preds)
- [ ] Add additional baseline datasets -> implement dynamic datasets
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
- [ ] **Average prediction error over time + average gradient over time**
- [ ] Scatter plot of absolute/relative error vs. inference time comparing surrogates.


