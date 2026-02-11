"""
    A minimal script for performing HyPE inference.
"""

import os
import sys
import torch
from transformers import CLIPTokenizer
from HySAC.hysac.models import HySAC
from HyperbolicSVDD.source.SVDD import LorentzHyperbolicOriginSVDD, project_to_lorentz

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LorentzHyperbolicOriginSVDD(curvature=2.3026, radius_lr=0.2, nu=0.01, center_init="origin")
model.load("./best_hyperbolic_svdd_model.pth")
model.center = model.center.to(DEVICE)


hyperbolic_clip = HySAC.from_pretrained("aimagelab/hysac", device=DEVICE).to(DEVICE)
hyperbolic_clip.eval()
clip_model_name = "openai/clip-vit-large-patch14"
tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)

prompt = "Several birds that are flying together over a frozen lake."

input_ids = tokenizer(prompt, return_tensors='pt', padding="max_length", truncation=True,
                      max_length=77).input_ids.to(DEVICE)

embedding = hyperbolic_clip.encode_text(input_ids, project=True)
embedding = project_to_lorentz(embedding, model.curvature)

prediction = model.predict(embedding)
print("Prediction:", prediction)  # Output: 0 for harmful, 1 for benign
