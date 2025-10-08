# %%
import torch
import numpy as np
from SVDD import *
import pandas as pd


# This function implements grid search for the nu parameter
def grid_search_nu(
    train_points,
    validation_path,
    nu_values=[0.01, 0.05, 0.1, 0.2, 0.3],
    curvature=2.3026,
    seed=42,
    epochs=20,
    center_init="origin",
    early_stopping_patience=10,
):
    """
    Perform grid search for the best nu parameter.

    Parameters:
    -----------
    train_points : torch.Tensor
        Training data points
    validation_path : str
        Path to validation data
    nu_values : list
        List of nu values to try
    curvature : float
        Curvature of the hyperbolic space
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
    results = []
    best_val_score = float("inf")
    best_model = None
    best_nu = None

    # Load validation data once to evaluate final models
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
        val_loss = None
        if len(benign_validation) > 0:
            val_loss, val_predictions = model.evaluate(benign_validation)
            benign_accuracy = torch.sum(val_predictions).item() / len(benign_validation)
        else:
            benign_accuracy = None

        # Evaluate on malicious validation samples (if available)
        malicious_accuracy = None
        f1_score = None
        if len(malicious_validation) > 0:
            malicious_predictions = model.predict(malicious_validation)
            # For malicious samples, prediction=0 means correctly identified as malicious (outside boundary)
            malicious_accuracy = torch.sum(malicious_predictions == 0).item() / len(
                malicious_validation
            )

            # Calculate F1 score if we have both benign and malicious samples
            if benign_accuracy is not None:
                # True positives: malicious correctly identified as malicious (prediction=0)
                TP = torch.sum(malicious_predictions == 0).item()
                # False positives: benign incorrectly identified as malicious (prediction=0)
                FP = (
                    torch.sum(val_predictions == 0).item()
                    if len(benign_validation) > 0
                    else 0
                )
                # False negatives: malicious incorrectly identified as benign (prediction=1)
                FN = torch.sum(malicious_predictions == 1).item()

                precision = TP / (TP + FP) if (TP + FP) > 0 else 0
                recall = TP / (TP + FN) if (TP + FN) > 0 else 0
                f1_score = (
                    2 * (precision * recall) / (precision + recall)
                    if (precision + recall) > 0
                    else 0
                )

        # Store results
        result = {
            "nu": nu,
            "radius": model.radius_param.item(),
            "best_radius": model.best_radius,
            "validation_loss": val_loss,
            "benign_accuracy": benign_accuracy,
            "malicious_accuracy": malicious_accuracy,
            "f1_score": f1_score,
        }
        results.append(result)

        # Update best model if this one is better (using F1 score if available, otherwise validation loss)
        score_metric = (
            -f1_score if f1_score is not None else val_loss
        )  # negative F1 because we're minimizing
        if score_metric is not None and score_metric < best_val_score:
            best_val_score = score_metric
            best_model = model
            best_nu = nu
            print(
                f"New best model with nu={nu}, score={-score_metric if f1_score is not None else score_metric:.4f}"
            )

    # Print results table
    print("\nGrid Search Results:")
    print(
        f"{'nu':^10} | {'Radius':^10} | {'Val Loss':^10} | {'Benign Acc':^10} | {'Malicious Acc':^10} | {'F1 Score':^10}"
    )
    print("-" * 70)

    for result in results:
        nu = result["nu"]
        radius = result["radius"]
        val_loss = result["validation_loss"]
        benign_acc = result["benign_accuracy"]
        malicious_acc = result["malicious_accuracy"]
        f1 = result["f1_score"]

        print(
            f"{nu:^10.3f} | {radius:^10.3f} | {val_loss if val_loss else 'N/A':^10} | "
            f"{benign_acc*100 if benign_acc else 'N/A':^10.2f}% | "
            f"{malicious_acc*100 if malicious_acc else 'N/A':^10.2f}% | "
            f"{f1*100 if f1 else 'N/A':^10.2f}%"
        )

    print(
        f"\nBest nu value: {best_nu} with {'F1 score' if f1_score is not None else 'validation loss'}: "
        f"{-best_val_score if f1_score is not None else best_val_score:.4f}"
    )

    # write all the results to a csv file
    import csv

    with open("grid_search_results.csv", mode="w", newline="") as file:
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
                    (
                        result["validation_loss"]
                        if result["validation_loss"] is not None
                        else "N/A"
                    ),
                    (
                        result["benign_accuracy"]
                        if result["benign_accuracy"] is not None
                        else "N/A"
                    ),
                    (
                        result["malicious_accuracy"]
                        if result["malicious_accuracy"] is not None
                        else "N/A"
                    ),
                    result["f1_score"] if result["f1_score"] is not None else "N/A",
                ]
            )

    return best_model, best_nu, results

def parse_validation_visu_dataset():
    val_visu_path = "./visu_text_validation.csv"

    ds = pd.read_csv(val_visu_path)
    prompts = list(ds["nsfw"])
    categories = ["malicious"] * len(prompts)
    # Load the dataset from the CSV file - benign prompts
    benign_prompts = list(ds["safe"])
    prompts.extend(benign_prompts)
    categories.extend(["benign"] * len(benign_prompts))
    return prompts, categories
    
def parse_train_visu_dataset():
    train_visu_path = "./visu_text_train.csv"

    ds = pd.read_csv(train_visu_path)

    prompts = list(ds["nsfw"])
    print("number of nsfw prompts :", len(prompts))
    categories = ["malicious"] * len(prompts)
    # Load the dataset from the CSV file - benign prompts

    benign_prompts = list(ds["safe"])
    print("number of sfw prompts :", len(benign_prompts))
    prompts.extend(benign_prompts)
    categories.extend(["benign"] * len(benign_prompts))
    return prompts, categories

if __name__ == "__main__":
    # first define the embeddings for visu dataset with HySAC
    hyperbolic_path = "./train_visu.pt"
    validation_path = "./validation_visu.pt"

    # Load training data
    hyperbolic_points = torch.load(hyperbolic_path)

    # Get benign training points
    benign_points = []
    for point in hyperbolic_points:
        if point[1] == "benign":
            benign_points.append(point[0])

    benign_points = torch.stack(benign_points)
    print(f"Number of benign training points: {benign_points.shape}")

    # Define nu values to try
    nu_values = np.linspace(0.1, 1, 20)

    # Run grid search
    best_model, best_nu, results = grid_search_nu(
        train_points=benign_points,
        validation_path=validation_path,
        nu_values=nu_values,
        curvature=2.3026,
        epochs=15,
        early_stopping_patience=5,
    )

    # Save best model
    best_model.save("best_hyperbolic_svdd_model.pth")
    print(
        f"Best model saved with nu={best_nu} and radius={best_model.radius_param.item():.4f}"
    )

    # Optional: Plot grid search results
    try:
        import matplotlib.pyplot as plt

        # Extract values for plotting
        nus = [r["nu"] for r in results]
        f1_scores = [r["f1_score"] for r in results if r["f1_score"] is not None]
        benign_accs = [
            r["benign_accuracy"] for r in results if r["benign_accuracy"] is not None
        ]
        malicious_accs = [
            r["malicious_accuracy"]
            for r in results
            if r["malicious_accuracy"] is not None
        ]

        # Only plot if we have results
        if len(f1_scores) > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(nus[: len(f1_scores)], f1_scores, "o-", label="F1 Score")
            plt.plot(
                nus[: len(benign_accs)], benign_accs, "s-", label="Benign Accuracy"
            )
            plt.plot(
                nus[: len(malicious_accs)],
                malicious_accs,
                "^-",
                label="Malicious Accuracy",
            )
            plt.xscale("log")
            plt.xlabel("nu value (log scale)")
            plt.ylabel("Score")
            plt.title("Grid Search Results for nu Parameter")
            plt.grid(True)
            plt.legend()
            plt.savefig("nu_grid_search_results.png")
            plt.show()
    except:
        print("Could not generate plot. Matplotlib may not be available.")
# %%

# Optional: Plot grid search results
try:
    import matplotlib.pyplot as plt
    import scienceplots

    plt.style.use("science")

    # Extract values for plotting
    nus = [r["nu"] for r in results]
    f1_scores = [r["f1_score"] for r in results if r["f1_score"] is not None]
    benign_accs = [
        r["benign_accuracy"] for r in results if r["benign_accuracy"] is not None
    ]
    malicious_accs = [
        r["malicious_accuracy"] for r in results if r["malicious_accuracy"] is not None
    ]

    # Only plot if we have results
    if len(f1_scores) > 0:
        plt.figure(figsize=(15, 9))
        plt.plot(
            nus[: len(f1_scores)], f1_scores, "o-", label="F1 Score", markersize=10
        )
        plt.plot(
            nus[: len(benign_accs)],
            benign_accs,
            "s-",
            label="Benign Accuracy",
            markersize=10,
        )
        plt.plot(
            nus[: len(malicious_accs)],
            malicious_accs,
            "^-",
            label="Malicious Accuracy",
            markersize=10,
        )
        plt.xlabel(r"$\nu$ value", fontsize=25)
        plt.grid(True)
        # set xticks to be the nu values, one each 2
        plt.xticks(nus[::2], fontsize=20)
        plt.yticks(fontsize=25)
        plt.legend()
        # increase legend font size
        plt.legend(fontsize=23)
        plt.savefig("nu_grid_search_results.png")
        plt.show()
except:
    print("Could not generate plot. Matplotlib may not be available.")
# %%
