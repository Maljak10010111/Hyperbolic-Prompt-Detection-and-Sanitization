# %%
import math
import torch
from torch import Tensor
from HyperbolicSVDD.notebooks.SVDD import *

import os
import pandas as pd
import torch
# %% 
benign_model = LorentzHyperbolicOriginSVDD(curvature=2.3026, radius_lr=0.2, nu=0.01, center_init="origin")
# load the best one
benign_model.load("best_hyperbolic_svdd_model.pth")

# %%
folder_path = "./EMBEDDINGS/hyperbolic_safe_clip"
embedding_dict = {}

for subfolders in os.listdir(folder_path):
    subfolder_path = os.path.join(folder_path, subfolders)
    if not subfolder_path.startswith('./EMBEDDINGS/hyperbolic_safe_clip/adv_MMA'):
        continue
    if os.path.isdir(subfolder_path):
        for file_name in os.listdir(subfolder_path):
            if file_name.endswith(".pt") and not "i2p" in file_name:

                print("*" * 20, file_name, "*" * 20)
                file_path = os.path.join(subfolder_path, file_name)
                embeddings = torch.load(file_path)
                embedding_dict[file_name] = embeddings
                clean_data, malicious_data = [], []
                for embedding in embeddings:
                    if "benign" == embedding[1]:
                        clean_data.append(project_to_lorentz(embedding[0], benign_model.curvature))
                    elif "malicious" == embedding[1]:
                        malicious_data.append(project_to_lorentz(embedding[0], benign_model.curvature))
                clean_predictions , mal_predictions = [], []
                for clean_d in clean_data:
                    clean_pred = benign_model.predict(clean_d.unsqueeze(0))
                    clean_predictions.append(clean_pred)
                for malicious_d in malicious_data:
                    malicious_pred = benign_model.predict(malicious_d.unsqueeze(0))
                    mal_predictions.append(malicious_pred)

                # compute True Positive and False Positive and True Negative and False negative
                TP = sum((torch.tensor(mal_predictions) == 0).tolist())
                FP = sum((torch.tensor(clean_predictions) == 0).tolist())
                TN = sum((torch.tensor(clean_predictions) == 1).tolist())
                FN = sum((torch.tensor(mal_predictions) == 1).tolist())

                clean_accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
                malicious_accuracy = TP / (TP + FN) if (TP + FN) > 0 else 0.0
                print(f"Clean accuracy: {clean_accuracy:.4f}")
                print(f"Malicious accuracy: {malicious_accuracy:.4f}")


                # compute precision, recall , F1 score
                precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
                recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
                F1_score = (
                    2 * (precision * recall) / (precision + recall)
                    if (precision + recall) > 0
                    else 0.0
                )

                # create a table to print that reports all the measures
                data = {
                    "Metric": [
                        "Accuracy",
                        "Precision",
                        "Recall",
                        "F1 Score",
                        "True Positives",
                        "False Positives",
                        "True Negatives",
                        "False Negatives",
                    ],
                    "Value": [
                        clean_accuracy,
                        precision,
                        recall,
                        F1_score,
                        TP,
                        FP,
                        TN,
                        FN,
                    ],
                }
                df = pd.DataFrame(data)
                print(df)
