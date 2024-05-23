from argparse import ArgumentParser

from benchmark.bench_fcts import run_benchmark, compare_models
from benchmark.bench_utils import check_surrogate_model


from utils import read_yaml_config


def main(args):
    conf = read_yaml_config(args.config)
    surrogates = conf["surrogates"]
    # Create dictionary to store metrics for each surrogate model
    metrics = {surrogate: {} for surrogate in surrogates}
    # Run benchmark for each surrogate model
    for model in surrogates:
        check_surrogate_model(model)
        metrics = run_benchmark(model, conf)
    # Compare models
    if conf["compare"]:
        compare_models(metrics)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", required=True)
    args = parser.parse_args()
    main(args)
