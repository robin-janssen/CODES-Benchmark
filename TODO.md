# Todos for the project

- [ ] **Write NeurIPS submission**
- [x] Read into DataHosting on HeiDATA -> use Zenodo for now
    - [x] Also upload other datasets and create download function
- [ ] Choose a code licence for the code - MIT?
- [x] Reach out to Lorenzo regarding the dataset
- [ ] Optuna and final run for all datasets.
- [ ] Setup repository: Copy over actions and other settings from the SSC Cookiecutter Template
- [ ] **Add links in the documentation to respective other sections**
- [x] Rename main scripts on the website (run_benchmark.py, run_training.py)

## Refactor
- [x] Remove the # model.n_timesteps = 100 lines in bench_fcts once new models are trained.
- [x] **Find a catchy name** - CODES Benchmark: Coupled ODE System Surrogates
- [x] Think about the organisation of the model config files
- [x] **Complete the README.md**
- [x] Rename NeuralODE to LatentNeuralODE
- [x] **Check the time benchmarking**
- [x] **Check whether the training is indeed deterministic.**
- [x] **Implement optional log scale labels in plots**
- [ ] **Check the memory benchmarking**
- [x] **Docstrings completion.**
- [x] **Make a clean requirements.txt**
- [ ] **Save Plots as SVG for immaculate quality**
- [ ] **Do a fresh test install of the repo and verify everything is running**
- [ ] **Check/update the config maker**
- [ ] **Add testing (for workshop)**
- [ ] Fix the task list for faulty runs in the benchmark
- [ ] Rename batch_scaling to batch.
- [ ] Think about the error plots - where should absolute errors be used, where relative errors? Does it make sense to use relative errors for the chemical error plots?
- [ ] Make tutorial notebooks/examples

## New Features
- [x] **Javascript config maker**
- [x] **Make a table in the CLI at the end of the benchmark with the most important metrics**
- [x] **Make a csv file with the most important metrics**
- [x] **Make a json file with the most important metrics**
- [x] **Add and document Lorenzo's Dataset**
- [x] Optuna tuning script
- [x] Calculate and add error quantities per surrogate to the individal (and comparative) outputs.
- [x] Add user prompt on whether to use task list or overwrite it.
- [x] Add additional baseline datasets -> implement dynamic datasets
- [ ] Time estimation using training duration of the main models an effective number of models to train.
- [ ] Store output in a .txt file
- [ ] Refactoring for more generality (remove chemistry specific code)
- [ ] Continue training of the existing models in case further convergence is needed
- [ ] Accumulation of error when predicting quantities iteratively
- [ ] Inter-/extrapolation in the initial conditions (domain shift)
- [ ] Dataset Visualizations - also include how the distribution changes over time (for iterative preds)
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
- [ ] Add the name of the surrogate and dataset to the error_dist per quantitiy and mention relative errors in title and axis labels
- [ ] Add "absolute" to the y axis of the uq heatmaps plot
- [ ] Change the figsize of the heatmap plots
- [ ] Contour plots to compare dynamics/UQ correlations
- [ ] **Average prediction error over time + average gradient over time**
- [ ] Scatter plot of absolute/relative error vs. inference time comparing surrogates.


