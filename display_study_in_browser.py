import optuna
from optuna.visualization import (
    plot_intermediate_values,
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_slice,
    plot_contour,
    plot_edf,
)

STUDY = "NeuralODE Optimization"
STORAGE = "sqlite:////export/data/isulzer/DON-vs-NODE/study/end_to_end.db"

study = optuna.load_study(study_name=STUDY, storage=STORAGE)

print("Best trial:")
print("  Value: ", study.best_trial.value)
print("  Params: ")
for key, value in study.best_trial.params.items():
    print(f"    {key}: {value}")

plot_optimization_history(study).show()
plot_param_importances(study).show()
plot_parallel_coordinate(study).show()
plot_intermediate_values(study).show()
plot_slice(study).show()
plot_contour(study).show()
plot_edf(study).show()
