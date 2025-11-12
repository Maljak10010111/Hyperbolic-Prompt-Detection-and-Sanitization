# Load model directly
from transformers import AutoImageProcessor, AutoModelForImageClassification
from diffusers import StableDiffusionPipeline
import torch
import os
from diffusers import DiffusionPipeline
import sys
from transformers import CLIPTokenizer
from HySAC.hysac.models import HySAC
from HySAC.hysac.lorentz import *

from geoopt.manifolds.lorentz import Lorentz
from torchmetrics.functional.multimodal import clip_score
from functools import partial
import csv
import pandas as pd


sys.path.append(
    "/mnt/ssd1/mary/Diffusion-Models-Embedding-Space-Defense/HyperbolicSVDD/notebooks"
)
# sys.path.append("/mnt/ssd1/mary/Diffusion-Models-Embedding-Space-Defense/HyperbolicSVDD/notebooks/SVDD_th.py")
from SVDD_th import LorentzHyperbolicOriginSVDD



class Config:
    """Configuration matching the training script"""

    CURVATURE_K = 2.3026
    NUM_FEATURES = 769
    NUM_CLASSES = 1
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")


def calculate_clip_score(images, prompts):
    images_int = (images * 255).astype("uint8")
    clip_score = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()
    return round(float(clip_score), 4)



def instantiate_classifier(model_path: str):
    """Load trained model with proper error handling"""
    try:
        print("Setting up models...")

        # Initialize HySAC model
        model_id = "aimagelab/hysac"
        hy_model = HySAC.from_pretrained(model_id, device=Config.DEVICE).to(
            Config.DEVICE
        )


        manifold = Lorentz(
            k=Config.CURVATURE_K 
        )
        print(f"Using Lorentz manifold with curvature k={manifold.k.item()}")

        svdd = LorentzHyperbolicOriginSVDD(
            curvature=2.3026, radius_lr=0.2, nu=0.01, center_init="origin"
        )
        # load the best one
        svdd.load(
            "/mnt/ssd1/mary/Diffusion-Models-Embedding-Space-Defense/HyperbolicSVDD/notebooks/best_hyperbolic_svdd_model.pth"
        )

        print(f"Models initialized on device: {Config.DEVICE}")
        print(f"Lorentz manifold curvature: {manifold.k.item()}")

        # Initialize tokenizer
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

        print(f"Successfully loaded model from: {model_path}")

        return hy_model, tokenizer, svdd
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e


def classify(classifier_model, prompt_embeddings):
    # print("Prompt embeddings shape:", prompt_embeddings.shape)
    if prompt_embeddings.shape[-1] != Config.NUM_FEATURES:
        time_prompt_embeddings = add_time_component(
            prompt_embeddings, torch.tensor(classifier_model.curvature)
        )
        # print("Time prompt embeddings shape:", time_prompt_embeddings.shape)

        is_valid, constraint_violation = validate_lorentz_embedding(
            time_prompt_embeddings,
            torch.tensor(classifier_model.curvature),
        )
        print(
            f"Is valid time prompt embeddings: {is_valid}, constraint violation: {constraint_violation}"
        )
        if is_valid:
            prompt_embeddings = time_prompt_embeddings


        logits = classifier_model.predict(time_prompt_embeddings)

        print("Logits:", logits)
        return logits, None
    else:
        print("No valid probabilities found.")
        return None, None


def validate_lorentz_embedding(embedding, manifold_k, tolerance=1e-5):
    """
    Validate that embedding satisfies Lorentz manifold constraint:
    -x0^2 + x1^2 + ... + xn^2 = -1/k
    """
    if embedding.dim() == 1:
        embedding = embedding.unsqueeze(0)
    # print("Embedding shape:", embedding.shape)
    #   print(embedding)
    # Compute the Lorentz inner product - manifold_k.abs() to ensure positive curvature
    lorentz_product = (
        torch.sum(embedding[0][1:] * embedding[0][1:], dim=-1)
        - embedding[0][0] * embedding[0][0]
    )
    # print("Lorentz product:", lorentz_product)

    # Expected value is -1 / |k|
    expected_value = -1.0 / manifold_k.abs().item()

    # Difference from expected hyperboloid constraint
    constraint_violation = torch.abs(lorentz_product - expected_value) + 1e-6
    # print("Constraint violation:", constraint_violation)

    is_valid = constraint_violation < tolerance

    return is_valid, constraint_violation


def add_time_component(data, manifold_k):
    expected_value = (
        -1 / manifold_k if manifold_k < 0 else 1 / manifold_k
    )  # Ensure positive curvature
    # print("Expected value for time component calculation:", expected_value)
    sum_var = torch.sum(data**2, dim=-1, keepdim=True)
    # print("Sum of squares of spatial components shape:", sum_var)
    x0 = torch.sqrt(expected_value + sum_var)
    # print("x0 shape:", x0)
    return torch.cat([x0, data], dim=-1)


# Alternative solution: import PIL if you want to convert numpy arrays
from PIL import Image as PILImage
import numpy as np

pipe = DiffusionPipeline.from_pretrained(
    "stablediffusionapi/newrealityxl-global-nsfw",
    torch_dtype=torch.float16,
    safety_checker=None,
).to("cuda")

# instanciate CLIP Vision text encoder 



svdd = LorentzHyperbolicOriginSVDD(
    curvature=Config.CURVATURE_K, radius_lr=0.2, nu=0.01, center_init="origin"
)
# load the best one
svdd.load(
    "/mnt/ssd1/mary/Diffusion-Models-Embedding-Space-Defense/HyperbolicSVDD/notebooks/best_hyperbolic_svdd_model.pth"
)
hy_model, tokenizer, svdd = instantiate_classifier('')

datasets_path = {
    "visu-original": "/mnt/ssd1/mary/Diffusion-Models-Embedding-Space-Defense/data/test_visu_text_prompts_paired.csv",

}
classes = {"normal": 0, "nsfw": 0}
dataset_classes = {
    "visu": classes.copy(),
}

for dataset_name, dataset_path in datasets_path.items():

    folder = f"./generated_images/{dataset_name}"
    os.makedirs(folder, exist_ok=True)
    # create an empty csv file to store the results
    cos_path = folder + f"/results_cos.csv"
    cos_header = ["idx", "NSFW:", "SFW:", "sd_clip_mal", "sd_clip_ben", "classification"]
    with open(cos_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(cos_header)

    df = pd.read_csv(dataset_path) if dataset_path.endswith(".csv") else pd.read_json(dataset_path, lines=True)
    print(f"Number of samples in {dataset_name}: {len(df)}")

    # Process each text prompt in the dataset
    for idx, row in df.iterrows():
        text_query = row["NSFW:"]
        # compute hyperbolic embedding 
        tokens = tokenizer(
            text_query, padding="max_length", max_length=77, return_tensors="pt"
        ).to("cuda")
        with torch.no_grad():
            text_embeddings = hy_model.encode_text(tokens["input_ids"], project=True)[0]
        text_embeddings = text_embeddings.unsqueeze(0)  # from [768] to [1,768]
        classification, confidence = classify(svdd, text_embeddings)

        # SOLUTION 2: Convert numpy array to PIL Image if using output_type="np"
        image = pipe(text_query, num_inference_steps=10, num_images_per_prompt=1,output_type="numpy").images
        sd_clip_mal = calculate_clip_score(image, [row["NSFW:"]])
        sd_clip_ben = calculate_clip_score(image, [row["SFW:"]])

        # write the results in the csv file, the delimiter is ';'
        with open(cos_path, "a") as f:
            writer = csv.writer(f)
            writer.writerow([idx, row["NSFW:"], row["SFW:"], sd_clip_mal, sd_clip_ben, classification if classification is not None else "N/A"])

        