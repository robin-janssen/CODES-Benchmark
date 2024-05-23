def run_benchmark(model, config):
    model_metrics = {}
    if config.accuracy:
        print(f"Running accuracy benchmark for {model}")
        # Run accuracy benchmark

    if config.dynamic:
        print(f"Running dynamic benchmark for {model}")
        # Run dynamic benchmark

    if config.timing:
        print(f"Running timing benchmark for {model}")
        # Run timing benchmark

    if config.interpolation:
        print(f"Running interpolation benchmark for {model}")
        # Run interpolation benchmark

    if config.extrapolation:
        print(f"Running extrapolation benchmark for {model}")
        # Run extrapolation benchmark

    if config.sparse:
        print(f"Running sparse benchmark for {model}")
        # Run sparse benchmark

    if config.UQ:
        print(f"Running UQ benchmark for {model}")
        # Run UQ benchmark

    return model_metrics


def compare_models(metrics):
    # Compare models
    pass
