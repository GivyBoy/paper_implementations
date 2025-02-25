import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import seaborn as sns
from tqdm import tqdm


def train_epoch(model, train_loader, optimizer, criterion, device, loss_scale_factor=1.0) -> Tuple[float, float]:
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        # Apply loss scaling for Figure 4 experiment
        loss = criterion(outputs, targets) * loss_scale_factor
        loss.backward()
        optimizer.step()

        # For tracking metrics, we use the original unscaled loss
        unscaled_loss = criterion(outputs, targets).item()
        total_loss += unscaled_loss
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return total_loss / len(train_loader), 100.0 * correct / total


def evaluate(model, test_loader, criterion, device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return total_loss / len(test_loader), 100.0 * correct / total


def plot_training_curves(results: Dict[str, Dict[str, List[float]]], title: str):
    """Plot training and test curves for multiple optimizers."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    for opt_name, metrics in results.items():
        ax1.plot(metrics["train_loss"], label=f"{opt_name} Train Loss")
        ax1.plot(metrics["test_loss"], label=f"{opt_name} Test Loss", linestyle="--")
        ax2.plot(metrics["train_acc"], label=f"{opt_name} Train Acc")
        ax2.plot(metrics["test_acc"], label=f"{opt_name} Test Acc", linestyle="--")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)
    ax1.set_title("Loss Curves")

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True)
    ax2.set_title("Accuracy Curves")

    plt.suptitle(title)
    plt.tight_layout()


def plot_heatmap(
    param_results: Dict[Tuple[float, float], float], lr_range: List[float], wd_range: List[float], title: str
):
    """Plot heatmap of final test accuracy for different hyperparameters."""
    results_matrix = np.zeros((len(lr_range), len(wd_range)))
    for i, lr in enumerate(lr_range):
        for j, wd in enumerate(wd_range):
            results_matrix[i, j] = param_results.get((lr, wd), 0)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        results_matrix,
        xticklabels=[f"{wd:.6f}" for wd in wd_range],
        yticklabels=[f"{lr:.6f}" for lr in lr_range],
        cmap="viridis",
        annot=True,
        fmt=".3f",
    )
    plt.xlabel("Weight Decay")
    plt.ylabel("Learning Rate")
    plt.title(title)
    plt.tight_layout()


def verify_gpu_usage(model, optimizer):
    """Verify all tensors are on GPU"""
    # Check model parameters
    for param in model.parameters():
        if not param.is_cuda:
            raise RuntimeError(f"Found model parameter on CPU: {param.shape}")

    # Check optimizer states
    for group in optimizer.param_groups:
        for p in group["params"]:
            if not p.is_cuda:
                raise RuntimeError("Found optimizer parameter on CPU")

            # Check optimizer state
            state = optimizer.state[p]
            for name, tensor in state.items():
                if isinstance(tensor, torch.Tensor) and not tensor.is_cuda:
                    raise RuntimeError(f"Found optimizer state {name} on CPU")

    print("All tensors are on GPU")


def print_gpu_memory():
    """Print GPU memory usage"""
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")


def plot_update_magnitudes_histogram(update_magnitudes, epoch, optimizer_name, bin_edges=None):
    """Plot histogram of parameter update magnitudes"""
    plt.figure(figsize=(8, 6))
    if bin_edges is None:
        # Use logarithmic bins from 2^-27 to 2^0
        bin_edges = np.logspace(-8.1, 0, 28)

    counts, edges = np.histogram(update_magnitudes, bins=bin_edges, density=True)
    plt.bar(range(len(counts)), counts, width=0.8)
    plt.xticks(range(len(counts)), [f"-{27-i}" for i in range(len(counts))])
    plt.xlabel("Order of magnitude (power of 2)")
    plt.ylabel("Proportion")
    plt.title(f"{optimizer_name} Epoch {epoch}")
    plt.tight_layout()

    return plt.gcf()


def collect_update_magnitudes(model, optimizer, alpha):
    """Collect magnitudes of parameter updates relative to the learning rate"""
    magnitudes = []

    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.grad is not None:
                state = optimizer.state[p]
                if "exp_avg" in state and "exp_avg_sq" in state:
                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    denom = exp_avg_sq.sqrt() + optimizer.defaults["eps"]
                    update = exp_avg / denom
                    magnitudes.extend((update.abs() / alpha).cpu().numpy().flatten())

    return np.array(magnitudes)


def train_and_evaluate(model_class, optimizer_class, hyperparams, epochs=300, use_bn=True, loss_scale_factor=1.0):
    """Train and evaluate a model with given optimizer and hyperparameters."""
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available! Training will be slow on CPU.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data loading and preprocessing
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    # Pin memory for faster GPU transfer
    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)

    train_loader = DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    results = {}
    histogram_data = {}
    torch.cuda.empty_cache()  # Clear GPU memory before training

    for lr, weight_decay in hyperparams:
        print(f"Training with lr={lr}, weight_decay={weight_decay}")
        # Create model and move to GPU immediately
        model = model_class(use_bn=use_bn)
        if torch.cuda.is_available():
            model = model.cuda()

        optimizer = optimizer_class(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Move optimizer states to GPU
        if torch.cuda.is_available():
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()

        criterion = nn.CrossEntropyLoss().to(device)

        # Verify GPU usage
        if torch.cuda.is_available():
            verify_gpu_usage(model, optimizer)
            print_gpu_memory()

        metrics = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
        update_magnitudes_by_epoch = {}

        for epoch in tqdm(range(epochs)):
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, criterion, device, loss_scale_factor=loss_scale_factor
            )
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)

            if epoch % 100 == 0 or epoch == epochs - 1:
                print(
                    f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%"
                )

                # Collect update magnitudes for visualization
                magnitudes = collect_update_magnitudes(model, optimizer, lr)
                update_magnitudes_by_epoch[epoch + 1] = magnitudes

            metrics["train_loss"].append(train_loss)
            metrics["train_acc"].append(train_acc)
            metrics["test_loss"].append(test_loss)
            metrics["test_acc"].append(test_acc)

        results[(lr, weight_decay)] = test_acc
        histogram_data[(lr, weight_decay)] = update_magnitudes_by_epoch

        # Clear memory after each hyperparameter combination
        del model, optimizer
        torch.cuda.empty_cache()

    return results, histogram_data


def train_compare_optimizers(
    model_class, optimizer_classes, best_hyperparams, epochs=300, use_bn=True, loss_scale_factor=1.0
):
    """Train and compare different optimizers using their best hyperparameters."""
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available! Training will be slow on CPU.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data loading and preprocessing
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    # Pin memory for faster GPU transfer
    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)

    train_loader = DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    results = {}
    histogram_data = {}

    for opt_name, opt_class in optimizer_classes.items():
        print(f"Training with {opt_name}")
        lr, weight_decay = best_hyperparams[opt_name]

        # Create model and move to GPU immediately
        model = model_class(use_bn=use_bn)
        if torch.cuda.is_available():
            model = model.cuda()

        optimizer = opt_class(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Move optimizer states to GPU
        if torch.cuda.is_available():
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()

        criterion = nn.CrossEntropyLoss().to(device)

        # Verify GPU usage
        if torch.cuda.is_available():
            verify_gpu_usage(model, optimizer)
            print_gpu_memory()

        metrics = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
        update_magnitudes_by_epoch = {}

        for epoch in tqdm(range(epochs)):
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, criterion, device, loss_scale_factor=loss_scale_factor
            )
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)

            if epoch % 100 == 0 or epoch == epochs - 1:
                print(
                    f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%"
                )

                # Collect update magnitudes for visualization
                magnitudes = collect_update_magnitudes(model, optimizer, lr)
                update_magnitudes_by_epoch[epoch + 1] = magnitudes

            metrics["train_loss"].append(train_loss)
            metrics["train_acc"].append(train_acc)
            metrics["test_loss"].append(test_loss)
            metrics["test_acc"].append(test_acc)

        results[opt_name] = metrics
        histogram_data[opt_name] = update_magnitudes_by_epoch

        # Clear memory after each optimizer
        del model, optimizer
        torch.cuda.empty_cache()

    return results, histogram_data
