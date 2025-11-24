# Load model directly
from transformers import AutoImageProcessor, AutoModelForImageClassification
from diffusers import StableDiffusionPipeline
import torch
import os
from diffusers import DiffusionPipeline
import sys
from transformers import CLIPTokenizer
import torch.nn.functional as F

from HySAC.hysac.models import HySAC
from HySAC.hysac.lorentz import *

from geoopt.manifolds.lorentz import Lorentz
from torchmetrics.functional.multimodal import clip_score
from functools import partial
import csv
import pandas as pd
import numpy as np


sys.path.append(
    "/mnt/ssd1/mary/Diffusion-Models-Embedding-Space-Defense/HyperbolicSVDD/notebooks"
)
# sys.path.append("/mnt/ssd1/mary/Diffusion-Models-Embedding-Space-Defense/HyperbolicSVDD/notebooks/SVDD_th.py")
from SVDD import LorentzHyperbolicOriginSVDD



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



def to_chw_tensor(x):
    # x: numpy array (1, H, W, 3)
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)          # → torch, still NHWC
    # Permute NHWC → NCHW
    x = x.permute(0, 3, 1, 2)            # → (1, 3, H, W)
    return x.to("cuda").float()          # keep exact pixel values



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
    "/mnt/ssd1/mary/Diffusion-Models-Embedding-Space-Defense/HyperbolicSVDD/notebooks/best_hyperbolic_svdd_model_paper.pth"
)
hy_model, tokenizer, svdd = instantiate_classifier('')

datasets_path = {
    "visu-original": "/mnt/ssd1/mary/Diffusion-Models-Embedding-Space-Defense/embeddings/data/visu_train.csv",
    "visu-test": "/mnt/ssd1/mary/Diffusion-Models-Embedding-Space-Defense/embeddings/data/visu_test.csv",
    "visu-val": "/mnt/ssd1/mary/Diffusion-Models-Embedding-Space-Defense/embeddings/data/visu_validation.csv",


}
classes = {"normal": 0, "nsfw": 0}
dataset_classes = {
    "visu": classes.copy(),
}

for dataset_name, dataset_path in datasets_path.items():

    folder = f"./generated_images_10k/{dataset_name}"
    os.makedirs(folder, exist_ok=True)
    # create an empty csv file to store the results
    cos_path = folder + f"/results_cos.csv"
    cos_header = ["idx", "NSFW:", "SFW:", "sd_clip_mal", "sd_clip_ben", "classification"]
    with open(cos_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(cos_header)

    df = pd.read_csv(dataset_path) if dataset_path.endswith(".csv") else pd.read_json(dataset_path, lines=True)
    print(f"Number of samples in {dataset_name}: {len(df)}")
    # check in the folder if there are already file that end for .pt, to continue from there
    existing_files = [f for f in os.listdir(folder) if f.endswith(".pt")]
    if existing_files:
        # count how many files are there
        num_existing_files = len(existing_files)
        num_prompt = num_existing_files * 100
        df = df.iloc[num_prompt:]
        print(f"Continuing from existing files, starting from prompt {num_prompt}")

    # Process each text prompt in the dataset
    all_image_embeddings = []
    for idx, row in df.iterrows():
        NSFW_text_query = row["NSFW"]
        SFW_text_query = row["SFW"]
        # compute hyperbolic embedding 
        NSFW_tokens = tokenizer(
            NSFW_text_query, padding="max_length", max_length=77, return_tensors="pt", truncation=True
        ).to("cuda")
        SFW_tokens = tokenizer(
            SFW_text_query, padding="max_length", max_length=77, return_tensors="pt", truncation=True
        ).to("cuda")
        with torch.no_grad():
            NSFW_text_embeddings = hy_model.encode_text(NSFW_tokens["input_ids"], project=True)[0]
            SFW_text_embeddings = hy_model.encode_text(SFW_tokens["input_ids"], project=True)[0]

        # SOLUTION 2: Convert numpy array to PIL Image if using output_type="np"
        NSFW_image = pipe(NSFW_text_query, num_inference_steps=10, num_images_per_prompt=1,output_type="numpy").images
        SFW_image = pipe(SFW_text_query, num_inference_steps=10, num_images_per_prompt=1,output_type="numpy").images
        print(NSFW_image.shape)
        # encode the images with hysac
        NSFW_img_tensor = to_chw_tensor(NSFW_image)
        SFW_img_tensor  = to_chw_tensor(SFW_image)
        NSFW_img_tensor = F.interpolate(NSFW_img_tensor, size=(224, 224), mode='bilinear')
        SFW_img_tensor = F.interpolate(SFW_img_tensor, size=(224, 224), mode='bilinear')

        with torch.no_grad():
            NSFW_image_embeddings = hy_model.encode_image(NSFW_img_tensor, project=True)[0]
            SFW_image_embeddings = hy_model.encode_image(SFW_img_tensor, project=True)[0]
            print("NSFW image embeddings shape:", NSFW_image_embeddings.shape)
            print("SFW image embeddings shape:", SFW_image_embeddings.shape)
        all_image_embeddings.append((NSFW_image_embeddings.cpu(), SFW_image_embeddings.cpu()))

        if len(all_image_embeddings) >= 100:
            # save the embeddings every 100 samples
            embeddings_path = folder + f"/image_embeddings_{idx}.pt"
            torch.save(all_image_embeddings, embeddings_path)
            all_image_embeddings = []
        # stop when the number of files in the folder is 10000
        if len(os.listdir(folder)) >= 10000:
            exit()
        


        