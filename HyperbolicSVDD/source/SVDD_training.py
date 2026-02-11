# %%
import torch
import numpy as np
import pandas as pd
import csv
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from SVDD import LorentzHyperbolicOriginSVDD, project_to_lorentz


def load_validation_data(
    validation_path: str, curvature: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load and preprocess validation data.
    
    Parameters:
    -----------
    validation_path : str
        Path to validation data
    curvature : float
        Curvature of the hyperbolic space
        
    Returns:
    --------
    benign_validation : torch.Tensor
        Benign validation samples
    malicious_validation : torch.Tensor
        Malicious validation samples
    """
    validation_data = torch.load(validation_path)
    benign_validation = []
    malicious_validation = []

    for point in validation_data:
        if point[1] == "benign":
            benign_validation.append(point[0])
        elif point[1] == "malicious":
            malicious_validation.append(point[0])

    benign_validation = (
        torch.stack(benign_validation) if benign_validation else torch.tensor([])
    )
    malicious_validation = (
        torch.stack(malicious_validation) if malicious_validation else torch.tensor([])
    )

    if len(benign_validation) > 0:
        benign_validation = project_to_lorentz(benign_validation, curvature)
    if len(malicious_validation) > 0:
        malicious_validation = project_to_lorentz(malicious_validation, curvature)

    return benign_validation, malicious_validation


def evaluate_model_on_validation(
    model: LorentzHyperbolicOriginSVDD,
    benign_validation: torch.Tensor,
    malicious_validation: torch.Tensor,
) -> Dict[str, Optional[float]]:
    """Evaluate model on validation data and compute metrics.
    
    Parameters:
    -----------
    model : LorentzHyperbolicOriginSVDD
        Trained SVDD model
    benign_validation : torch.Tensor
        Benign validation samples
    malicious_validation : torch.Tensor
        Malicious validation samples
        
    Returns:
    --------
    metrics : dict
        Dictionary containing validation loss, accuracies, and F1 score
    """
    metrics = {
        "validation_loss": None,
        "benign_accuracy": None,
        "malicious_accuracy": None,
        "f1_score": None,
    }

    # Evaluate on benign samples
    if len(benign_validation) > 0:
        val_loss, val_predictions = model.evaluate(benign_validation)
        metrics["validation_loss"] = val_loss
        metrics["benign_accuracy"] = (
            torch.sum(val_predictions).item() / len(benign_validation)
        )
    else:
        val_predictions = None

    # Evaluate on malicious samples
    if len(malicious_validation) > 0:
        malicious_predictions = model.predict(malicious_validation)
        metrics["malicious_accuracy"] = (
            torch.sum(malicious_predictions == 0).item() / len(malicious_validation)
        )

        # Calculate F1 score if we have both benign and malicious samples
        if metrics["benign_accuracy"] is not None:
            TP = torch.sum(malicious_predictions == 0).item()
            FP = (
                torch.sum(val_predictions == 0).item()
                if val_predictions is not None
                else 0
            )
            FN = torch.sum(malicious_predictions == 1).item()

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            metrics["f1_score"] = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

    return metrics


def save_grid_search_results(results: List[Dict], output_path: str = "grid_search_results.csv") -> None:
    """Save grid search results to CSV file.
    
    Parameters:
    -----------
    results : list
        List of dictionaries with results for each nu value
    output_path : str
        Path to save the CSV file
    """
    with open(output_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "nu",
                "radius",
                "best_radius",
                "validation_loss",
                "benign_accuracy",
                "malicious_accuracy",
                "f1_score",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result["nu"],
                    result["radius"],
                    result["best_radius"],
                    result["validation_loss"] if result["validation_loss"] is not None else "N/A",
                    result["benign_accuracy"] if result["benign_accuracy"] is not None else "N/A",
                    result["malicious_accuracy"] if result["malicious_accuracy"] is not None else "N/A",
                    result["f1_score"] if result["f1_score"] is not None else "N/A",
                ]
            )


def print_grid_search_results(results: List[Dict], best_nu: float, best_val_score: float) -> None:
    """Print formatted grid search results.
    
    Parameters:
    -----------
    results : list
        List of dictionaries with results for each nu value
    best_nu : float
        The best nu value found
    best_val_score : float
        The best validation score achieved
    """
    print("\nGrid Search Results:")
    print(
        f"{'nu':^10} | {'Radius':^10} | {'Val Loss':^10} | {'Benign Acc':^10} | {'Malicious Acc':^10} | {'F1 Score':^10}"
    )
    print("-" * 70)

    for result in results:
        val_loss = result["validation_loss"]
        benign_acc = result["benign_accuracy"]
        malicious_acc = result["malicious_accuracy"]
        f1 = result["f1_score"]

        print(
            f"{result['nu']:^10.3f} | {result['radius']:^10.3f} | "
            f"{val_loss if val_loss else 'N/A':^10} | "
            f"{benign_acc*100 if benign_acc else 'N/A':^10.2f}% | "
            f"{malicious_acc*100 if malicious_acc else 'N/A':^10.2f}% | "
            f"{f1*100 if f1 else 'N/A':^10.2f}%"
        )

    has_f1 = any(r["f1_score"] is not None for r in results)
    metric_name = "F1 score" if has_f1 else "validation loss"
    score_value = -best_val_score if has_f1 else best_val_score
    print(f"\nBest nu value: {best_nu} with {metric_name}: {score_value:.4f}")


def grid_search_nu(
    train_points: torch.Tensor,
    validation_path: str,
    nu_values: List[float] = None,
    curvature: float = 2.3026,
    seed: int = 42,
    epochs: int = 20,
    center_init: str = "origin",
    early_stopping_patience: int = 10,
) -> Tuple[LorentzHyperbolicOriginSVDD, float, List[Dict]]:
    """
    Perform grid search for the best nu parameter.

    Parameters:
    -----------
    train_points : torch.Tensor
        Training data points
    validation_path : str
        Path to validation data
    nu_values : list, optional
        List of nu values to try (default: [0.01, 0.05, 0.1, 0.2, 0.3])
    curvature : float
        Curvature of the hyperbolic space
    seed : int
        Random seed for reproducibility
    epochs : int
        Number of training epochs for each model
    center_init : str
        Method to initialize the center
    early_stopping_patience : int
        Number of epochs to wait before early stopping

    Returns:
    --------
    best_model : LorentzHyperbolicOriginSVDD
        The best model found during grid search
    best_nu : float
        The best nu value found
    results : list
        List of dictionaries with results for each nu value
    """
    if nu_values is None:
        nu_values = [0.01, 0.05, 0.1, 0.2, 0.3]
    
    results = []
    best_val_score = float("inf")
    best_model = None
    best_nu = None

    # Load validation data once
    benign_validation, malicious_validation = load_validation_data(validation_path, curvature)
    print(f"Grid Search: Testing {len(nu_values)} nu values: {nu_values}")

    for nu in nu_values:
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f"\n{'='*50}")
        print(f"Training model with nu = {nu}")
        print(f"{'='*50}")

        # Train model with current nu value
        model = LorentzHyperbolicOriginSVDD(
            curvature=curvature, radius_lr=0.2, nu=nu, center_init=center_init
        )

        model.fit(
            train_points,
            epochs=epochs,
            validation_path=validation_path,
            early_stopping_patience=early_stopping_patience,
        )

        # Evaluate on validation set
        metrics = evaluate_model_on_validation(model, benign_validation, malicious_validation)

        # Store results
        result = {
            "nu": nu,
            "radius": model.radius_param.item(),
            "best_radius": model.best_radius,
            "validation_loss": metrics["validation_loss"],
            "benign_accuracy": metrics["benign_accuracy"],
            "malicious_accuracy": metrics["malicious_accuracy"],
            "f1_score": metrics["f1_score"],
        }
        results.append(result)

        # Update best model if this one is better
        f1_score = metrics["f1_score"]
        val_loss = metrics["validation_loss"]
        score_metric = -f1_score if f1_score is not None else val_loss
        
        if score_metric is not None and score_metric < best_val_score:
            best_val_score = score_metric
            best_model = model
            best_nu = nu
            score_value = -score_metric if f1_score is not None else score_metric
            print(f"New best model with nu={nu}, score={score_value:.4f}")

    # Print and save results
    print_grid_search_results(results, best_nu, best_val_score)
    save_grid_search_results(results)

    return best_model, best_nu, results

def parse_visu_dataset(csv_path: str, verbose: bool = False) -> Tuple[List[str], List[str]]:
    """Parse VISU dataset from CSV file.
    
    Parameters:
    -----------
    csv_path : str
        Path to the CSV file containing prompts
    verbose : bool
        Whether to print dataset statistics
        
    Returns:
    --------
    prompts : list
        List of all prompts (malicious + benign)
    categories : list
        List of categories for each prompt
    """
    ds = pd.read_csv(csv_path)
    
    # Load malicious prompts
    malicious_prompts = list(ds["nsfw"])
    if verbose:
        print(f"Number of NSFW prompts: {len(malicious_prompts)}")
    
    # Load benign prompts
    benign_prompts = list(ds["safe"])
    if verbose:
        print(f"Number of safe prompts: {len(benign_prompts)}")
    
    # Combine prompts and categories
    prompts = malicious_prompts + benign_prompts
    categories = ["malicious"] * len(malicious_prompts) + ["benign"] * len(benign_prompts)
    
    return prompts, categories


def parse_validation_visu_dataset() -> Tuple[List[str], List[str]]:
    """Parse validation VISU dataset."""
    return parse_visu_dataset("./visu_text_validation.csv", verbose=False)


def parse_train_visu_dataset() -> Tuple[List[str], List[str]]:
    """Parse training VISU dataset."""
    return parse_visu_dataset("./visu_text_train.csv", verbose=True)

def load_benign_training_data(hyperbolic_path: str) -> torch.Tensor:
    """Load and filter benign training points.
    
    Parameters:
    -----------
    hyperbolic_path : str
        Path to hyperbolic embeddings
        
    Returns:
    --------
    benign_points : torch.Tensor
        Tensor of benign training points
    """
    hyperbolic_points = torch.load(hyperbolic_path)
    benign_points = [point[0] for point in hyperbolic_points if point[1] == "benign"]
    benign_points = torch.stack(benign_points)
    print(f"Number of benign training points: {benign_points.shape}")
    return benign_points


def plot_grid_search_results(results: List[Dict], output_path: str = "nu_grid_search_results.png") -> None:
    """Plot grid search results with science plots style.
    
    Parameters:
    -----------
    results : list
        List of dictionaries with results for each nu value
    output_path : str
        Path to save the plot
    """
    try:
        import matplotlib.pyplot as plt
        import scienceplots
        plt.style.use("science")
    except ImportError:
        print("Could not generate plot. Matplotlib or scienceplots may not be available.")
        return

    # Extract values for plotting
    nus = [r["nu"] for r in results]
    f1_scores = [r["f1_score"] for r in results if r["f1_score"] is not None]
    benign_accs = [r["benign_accuracy"] for r in results if r["benign_accuracy"] is not None]
    malicious_accs = [r["malicious_accuracy"] for r in results if r["malicious_accuracy"] is not None]

    if len(f1_scores) == 0:
        print("No F1 scores to plot.")
        return

    plt.figure(figsize=(15, 9))
    plt.plot(nus[:len(f1_scores)], f1_scores, "o-", label="F1 Score", markersize=10)
    plt.plot(nus[:len(benign_accs)], benign_accs, "s-", label="Benign Accuracy", markersize=10)
    plt.plot(nus[:len(malicious_accs)], malicious_accs, "^-", label="Malicious Accuracy", markersize=10)
    
    plt.xlabel(r"$\nu$ value", fontsize=25)
    plt.grid(True)
    plt.xticks(nus[::2], fontsize=20)
    plt.yticks(fontsize=25)
    plt.legend(fontsize=23)
    plt.savefig(output_path)
    plt.show()
    print(f"Plot saved to {output_path}")


def main():
    """Main function to run SVDD training with grid search."""
    # Configuration
    config = {
        "hyperbolic_path": "../../embeddings/hyperbolic_safe_clip/train_visu/train_visu.pt",
        "validation_path": "../../embeddings/hyperbolic_safe_clip/validation_visu/validation_visu.pt",
        "curvature": 2.3026,
        "epochs": 15,
        "early_stopping_patience": 5,
        "model_output_path": "best_hyperbolic_svdd_model.pth",
        "plot_output_path": "nu_grid_search_results.png",
    }
    
    # Load training data
    benign_points = load_benign_training_data(config["hyperbolic_path"])

    # Define nu values to try
    nu_values = np.linspace(0.01, 0.1, 20)

    # Run grid search
    best_model, best_nu, results = grid_search_nu(
        train_points=benign_points,
        validation_path=config["validation_path"],
        nu_values=nu_values,
        curvature=config["curvature"],
        epochs=config["epochs"],
        early_stopping_patience=config["early_stopping_patience"],
    )

    # Save best model
    best_model.save(config["model_output_path"])
    print(
        f"Best model saved to {config['model_output_path']} with "
        f"nu={best_nu} and radius={best_model.radius_param.item():.4f}"
    )
    
    # Plot results
    plot_grid_search_results(results, config["plot_output_path"])


if __name__ == "__main__":
    main()

# %%
