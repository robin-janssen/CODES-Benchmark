from argparse import ArgumentParser

from benchmark.bench_fcts import run_benchmark, compare_models
from benchmark.bench_utils import check_surrogate

from utils import read_yaml_config, nice_print
from surrogates.surrogate_classes import surrogate_classes


def main(args):

    config = read_yaml_config(args.config)
    surrogates = config["surrogates"]
    # Create dictionary to store metrics for all surrogate models
    all_metrics = {surrogate: {} for surrogate in surrogates}

    # Run benchmark for each surrogate model
    for surrogate_name in surrogates:
        if surrogate_name in surrogate_classes:
            nice_print(f"Running benchmark for {surrogate_name}")
            surrogate_class = surrogate_classes[surrogate_name]
            check_surrogate(surrogate_name, config)
            metrics = run_benchmark(surrogate_name, surrogate_class, config)
            all_metrics[surrogate_name] = metrics
        else:
            print(f"Surrogate {surrogate_name} not recognized. Skipping.")

    # Compare models
    if config["compare"]:
        if len(surrogates) < 2:
            nice_print("At least two surrogate models are required to compare.")
        else:
            nice_print("Comparing models")
            compare_models(all_metrics, config)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config", default="config.yaml", type=str, help="Path to the config file."
    )
    args = parser.parse_args()
    main(args)
