from .bench_fcts import (
    compare_batchsize,
    compare_dynamic_accuracy,
    compare_extrapolation,
    compare_inference_time,
    compare_interpolation,
    compare_main_losses,
    compare_models,
    compare_relative_errors,
    compare_sparse,
    compare_UQ,
    evaluate_accuracy,
    evaluate_batchsize,
    evaluate_compute,
    evaluate_dynamic_accuracy,
    evaluate_extrapolation,
    evaluate_interpolation,
    evaluate_sparse,
    evaluate_UQ,
    run_benchmark,
    tabular_comparison,
    time_inference,
)
from .bench_plots import (
    get_custom_palette,
    inference_time_bar_plot,
    int_ext_sparse,
    plot_all_generalization_errors,
    plot_average_errors_over_time,
    plot_average_uncertainty_over_time,
    plot_comparative_dynamic_correlation_heatmaps,
    plot_comparative_error_correlation_heatmaps,
    plot_dynamic_correlation,
    plot_dynamic_correlation_heatmap,
    plot_error_correlation_heatmap,
    plot_error_distribution_comparative,
    plot_error_distribution_per_quantity,
    plot_example_mode_predictions,
    plot_example_predictions_with_uncertainty,
    plot_generalization_error_comparison,
    plot_generalization_errors,
    plot_loss_comparison,
    plot_loss_comparison_train_duration,
    plot_losses,
    plot_MAE_comparison,
    plot_relative_errors,
    plot_relative_errors_over_time,
    plot_surr_losses,
    plot_uncertainty_confidence,
    plot_uncertainty_over_time_comparison,
    plot_uncertainty_vs_errors,
    rel_errors_and_uq,
    save_plot,
    save_plot_counter,
)
from .bench_utils import (
    check_benchmark,
    check_surrogate,
    clean_metrics,
    convert_dict_to_scientific_notation,
    convert_to_standard_types,
    count_trainable_parameters,
    discard_numpy_entries,
    flatten_dict,
    format_seconds,
    format_time,
    get_model_config,
    get_required_models_list,
    get_surrogate,
    load_model,
    make_comparison_csv,
    measure_memory_footprint,
    read_yaml_config,
    save_table_csv,
    write_metrics_to_yaml,
    measure_inference_time,
)

__all__ = [
    "run_benchmark",
    "evaluate_accuracy",
    "evaluate_dynamic_accuracy",
    "time_inference",
    "evaluate_compute",
    "evaluate_interpolation",
    "evaluate_extrapolation",
    "evaluate_sparse",
    "evaluate_batchsize",
    "evaluate_UQ",
    "compare_models",
    "compare_main_losses",
    "compare_relative_errors",
    "compare_inference_time",
    "compare_dynamic_accuracy",
    "compare_interpolation",
    "compare_extrapolation",
    "compare_sparse",
    "compare_batchsize",
    "compare_UQ",
    "tabular_comparison",
    "save_plot",
    "save_plot_counter",
    "plot_relative_errors_over_time",
    "plot_dynamic_correlation",
    "plot_generalization_errors",
    "plot_average_errors_over_time",
    "plot_example_predictions_with_uncertainty",
    "plot_example_mode_predictions",
    "plot_average_uncertainty_over_time",
    "plot_uncertainty_vs_errors",
    "plot_uncertainty_confidence",
    "plot_surr_losses",
    "plot_error_distribution_per_quantity",
    "plot_losses",
    "plot_loss_comparison",
    "plot_loss_comparison_train_duration",
    "plot_MAE_comparison",
    "plot_relative_errors",
    "plot_uncertainty_over_time_comparison",
    "inference_time_bar_plot",
    "plot_generalization_error_comparison",
    "plot_error_correlation_heatmap",
    "plot_dynamic_correlation_heatmap",
    "plot_error_distribution_comparative",
    "plot_comparative_error_correlation_heatmaps",
    "plot_comparative_dynamic_correlation_heatmaps",
    "get_custom_palette",
    "int_ext_sparse",
    "plot_all_generalization_errors",
    "rel_errors_and_uq",
    "check_surrogate",
    "check_benchmark",
    "get_required_models_list",
    "read_yaml_config",
    "load_model",
    "count_trainable_parameters",
    "measure_memory_footprint",
    "convert_to_standard_types",
    "discard_numpy_entries",
    "clean_metrics",
    "write_metrics_to_yaml",
    "get_surrogate",
    "format_time",
    "format_seconds",
    "flatten_dict",
    "convert_dict_to_scientific_notation",
    "make_comparison_csv",
    "save_table_csv",
    "get_model_config",
    "measure_inference_time",
]
