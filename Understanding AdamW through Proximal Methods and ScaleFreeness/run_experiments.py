from optimizers import AdamW, AdamL2, AdamProx, AdamProxL2
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from train_and_plot import (
    train_and_evaluate,
    plot_heatmap,
    train_compare_optimizers,
    plot_training_curves,
    plot_update_magnitudes_histogram,
)
from model import CNN


def run_figure2_experiments():
    """Run experiments for Figure 2 - with batch normalization"""
    print("Running experiments for Figure 2 - Comparison with Batch Normalization")

    # Define hyperparameter ranges
    lr_range = [0.00005, 0.0001, 0.0005, 0.001, 0.005]
    wd_range = [0, 0.00001, 0.00005, 0.0001, 0.0005, 0.001]

    hyperparams = [(lr, wd) for lr in lr_range for wd in wd_range]

    # Experiment configurations for CNN model with varying depth
    configs = [
        {"name": "CNN with BN (A)", "model_class": CNN, "use_bn": True, "epochs": 300},
        {"name": "CNN with BN (B)", "model_class": CNN, "use_bn": True, "epochs": 300},
    ]

    for config in configs:
        print(f"\nRunning experiments for {config['name']}")

        # Train with AdamW
        adamw_results, _ = train_and_evaluate(
            config["model_class"], AdamW, hyperparams, epochs=config["epochs"], use_bn=config["use_bn"]
        )

        # Train with Adam-L2
        adaml2_results, _ = train_and_evaluate(
            config["model_class"], AdamL2, hyperparams, epochs=config["epochs"], use_bn=config["use_bn"]
        )

        # Plot heatmaps
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 2, 1)
        plot_heatmap(adaml2_results, lr_range, wd_range, f"Adam-L2 {config['name']}")

        plt.subplot(1, 2, 2)
        plot_heatmap(adamw_results, lr_range, wd_range, f"AdamW {config['name']}")

        plt.savefig(f"figure2_{config['name'].replace(' ', '_')}.png")
        plt.close()


def run_figure3_experiments():
    """Run experiments for Figure 3 - without batch normalization"""
    print("Running experiments for Figure 3 - Comparison without Batch Normalization")

    # Define hyperparameter ranges
    lr_range = [0.00005, 0.0001, 0.0005, 0.001, 0.005]
    wd_range = [0, 0.00001, 0.00005, 0.0001, 0.0005, 0.001]

    hyperparams = [(lr, wd) for lr in lr_range for wd in wd_range]

    # Experiment configurations for CNN model without BN
    configs = [
        {"name": "CNN without BN (A)", "model_class": CNN, "use_bn": False, "epochs": 300},
        {"name": "CNN without BN (B)", "model_class": CNN, "use_bn": False, "epochs": 300},
    ]

    for config in configs:
        print(f"\nRunning experiments for {config['name']}")

        # Train with AdamW
        adamw_results, adamw_hist = train_and_evaluate(
            config["model_class"], AdamW, hyperparams, epochs=config["epochs"], use_bn=config["use_bn"]
        )

        # Train with Adam-L2
        adaml2_results, adaml2_hist = train_and_evaluate(
            config["model_class"], AdamL2, hyperparams, epochs=config["epochs"], use_bn=config["use_bn"]
        )

        # Find best hyperparameters for both optimizers
        best_adamw_lr, best_adamw_wd = max(adamw_results.items(), key=lambda x: x[1])[0]
        best_adaml2_lr, best_adaml2_wd = max(adaml2_results.items(), key=lambda x: x[1])[0]

        # Plot heatmaps
        plt.figure(figsize=(20, 5))

        plt.subplot(1, 4, 1)
        plot_heatmap(adaml2_results, lr_range, wd_range, f"Adam-L2 {config['name']}")

        plt.subplot(1, 4, 2)
        plot_heatmap(adamw_results, lr_range, wd_range, f"AdamW {config['name']}")

        # Run detailed comparison with best hyperparameters
        optimizer_classes = {
            "Adam-L2": AdamL2,
            "AdamW": AdamW,
        }
        best_hyperparams = {
            "Adam-L2": (best_adaml2_lr, best_adaml2_wd),
            "AdamW": (best_adamw_lr, best_adamw_wd),
        }

        detailed_results, detailed_hist = train_compare_optimizers(
            config["model_class"], optimizer_classes, best_hyperparams, epochs=config["epochs"], use_bn=config["use_bn"]
        )

        # Plot training curves
        plt.subplot(1, 4, 3)
        plt.plot(detailed_results["Adam-L2"]["train_loss"], label="Adam-L2 Train")
        plt.plot(detailed_results["Adam-L2"]["test_loss"], label="Adam-L2 Test", linestyle="--")
        plt.plot(detailed_results["AdamW"]["train_loss"], label="AdamW Train")
        plt.plot(detailed_results["AdamW"]["test_loss"], label="AdamW Test", linestyle="--")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss Curves")

        plt.subplot(1, 4, 4)
        plt.plot(detailed_results["Adam-L2"]["train_acc"], label="Adam-L2 Train")
        plt.plot(detailed_results["Adam-L2"]["test_acc"], label="Adam-L2 Test", linestyle="--")
        plt.plot(detailed_results["AdamW"]["train_acc"], label="AdamW Train")
        plt.plot(detailed_results["AdamW"]["test_acc"], label="AdamW Test", linestyle="--")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.title("Accuracy Curves")

        plt.tight_layout()
        plt.savefig(f"figure3_{config['name'].replace(' ', '_')}_performance.png")
        plt.close()

        # Plot update magnitude histograms at final epoch
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        adaml2_magnitudes = detailed_hist["Adam-L2"][config["epochs"]]
        plot_update_magnitudes_histogram(adaml2_magnitudes, config["epochs"], "Adam-L2")

        plt.subplot(1, 2, 2)
        adamw_magnitudes = detailed_hist["AdamW"][config["epochs"]]
        plot_update_magnitudes_histogram(adamw_magnitudes, config["epochs"], "AdamW")

        plt.tight_layout()
        plt.savefig(f"figure3_{config['name'].replace(' ', '_')}_histograms.png")
        plt.close()


def run_figure4_experiments():
    """Run experiments for Figure 4 - Loss scaling"""
    print("Running experiments for Figure 4 - Loss scaling comparison")

    # Define hyperparameter ranges
    lr_range = [0.00005, 0.0001, 0.0005, 0.001, 0.005]
    wd_range = [0, 0.00001, 0.00005, 0.0001, 0.0005, 0.001]

    hyperparams = [(lr, wd) for lr in lr_range for wd in wd_range]

    # Loss scaling factors to test
    scaling_factors = [1.0, 10.0, 100.0]

    # Use CNN without BN for this experiment
    model_class = CNN
    use_bn = False
    epochs = 300

    for scale in scaling_factors:
        print(f"\nRunning experiments with loss scaling factor {scale}")

        # Train with AdamW
        adamw_results, _ = train_and_evaluate(
            model_class, AdamW, hyperparams, epochs=epochs, use_bn=use_bn, loss_scale_factor=scale
        )

        # Train with Adam-L2
        adaml2_results, _ = train_and_evaluate(
            model_class, AdamL2, hyperparams, epochs=epochs, use_bn=use_bn, loss_scale_factor=scale
        )

        # Plot heatmaps
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 2, 1)
        plot_heatmap(adaml2_results, lr_range, wd_range, f"Adam-L2 (Loss Scale {scale})")

        plt.subplot(1, 2, 2)
        plot_heatmap(adamw_results, lr_range, wd_range, f"AdamW (Loss Scale {scale})")

        plt.savefig(f"figure4_loss_scale_{scale}.png")
        plt.close()


def run_figure5_experiments():
    """Run experiments for Figure 5 - AdamW vs AdamProx"""
    print("Running experiments for Figure 5 - AdamW vs AdamProx")

    # Define hyperparameter ranges
    lr_range = [0.00005, 0.0001, 0.0005, 0.001, 0.005]
    wd_range = [0, 0.00001, 0.00005, 0.0001, 0.0005, 0.001]

    hyperparams = [(lr, wd) for lr in lr_range for wd in wd_range]

    # Experiment configurations
    configs = [
        {"name": "CNN without BN", "model_class": CNN, "use_bn": False, "epochs": 300},
    ]

    for config in configs:
        print(f"\nRunning experiments for {config['name']}")

        # Train with AdamW
        adamw_results, _ = train_and_evaluate(
            config["model_class"], AdamW, hyperparams, epochs=config["epochs"], use_bn=config["use_bn"]
        )

        # Train with AdamProx
        adamprox_results, _ = train_and_evaluate(
            config["model_class"], AdamProx, hyperparams, epochs=config["epochs"], use_bn=config["use_bn"]
        )

        # Plot heatmaps
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 2, 1)
        plot_heatmap(adamw_results, lr_range, wd_range, f"AdamW {config['name']}")

        plt.subplot(1, 2, 2)
        plot_heatmap(adamprox_results, lr_range, wd_range, f"AdamProx {config['name']}")

        plt.savefig(f"figure5_{config['name'].replace(' ', '_')}.png")
        plt.close()


def run_figure6_experiments():
    """Run experiments for Figure 6 - AdamW vs AdamProxL2"""
    print("Running experiments for Figure 6 - AdamW vs AdamProxL2")

    # Define hyperparameter ranges
    lr_range = [0.00005, 0.0001, 0.0005, 0.001, 0.005]
    wd_range = [0, 0.00001, 0.00005, 0.0001, 0.0005, 0.001]

    hyperparams = [(lr, wd) for lr in lr_range for wd in wd_range]

    # Use CNN without BN for this experiment
    model_class = CNN
    use_bn = False
    epochs = 300

    print("\nRunning AdamW vs AdamProxL2 comparison")

    # Train with AdamW
    adamw_results, _ = train_and_evaluate(model_class, AdamW, hyperparams, epochs=epochs, use_bn=use_bn)

    # Train with AdamProxL2
    adamproxl2_results, _ = train_and_evaluate(model_class, AdamProxL2, hyperparams, epochs=epochs, use_bn=use_bn)

    # Plot heatmaps
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plot_heatmap(adamw_results, lr_range, wd_range, "AdamW")

    plt.subplot(1, 2, 2)
    plot_heatmap(adamproxl2_results, lr_range, wd_range, "AdamProxL2")

    plt.savefig("figure6_adamw_vs_adamproxl2.png")
    plt.close()


def main():
    # Choose which experiments to run
    run_figure2 = True  # With BN
    run_figure3 = True  # Without BN
    run_figure4 = True  # Loss scaling
    run_figure5 = True  # AdamW vs AdamProx
    run_figure6 = True  # AdamW vs AdamProxL2

    if run_figure2:
        run_figure2_experiments()

    if run_figure3:
        run_figure3_experiments()

    if run_figure4:
        run_figure4_experiments()

    if run_figure5:
        run_figure5_experiments()

    if run_figure6:
        run_figure6_experiments()


if __name__ == "__main__":
    main()
