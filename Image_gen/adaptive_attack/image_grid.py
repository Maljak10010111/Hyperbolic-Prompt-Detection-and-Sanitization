import os
import pandas as pd
from diffusers import DiffusionPipeline
import torch
from PIL import Image as PILImage
import numpy as np

pipe = DiffusionPipeline.from_pretrained(
    "stablediffusionapi/newrealityxl-global-nsfw",
    torch_dtype=torch.float16,
    safety_checker=None,
).to("cuda")

num_lines = 7
current_dir = os.path.dirname(os.path.abspath(__file__))
target_img_count = 0
indexes = [10, 5,12,14, 0] 
print(indexes)
# iterate in all the subfolders of the current directory
for root, dirs, files in os.walk(current_dir):
    for dir_name in dirs:
        if '00' in dir_name:
            dir_path = os.path.join(root, dir_name)

            # look for a csv file that has 'results' in its name
            for file_name in os.listdir(dir_path):
                if "results" in file_name:
                    file_path = os.path.join(dir_path, file_name)
                    # load the file in a dataframe and print the first num_lines lines

                    df = pd.read_csv(file_path)
                    for idx in indexes:
                        prompt = df.iloc[idx]["target_prompt"]
                        print(f"Generating image for prompt: {prompt}")
                        images = pipe(prompt, num_inference_steps=20, guidance_scale=7.5).images
                        for img in images:
                            img = img.resize((512, 512), PILImage.LANCZOS)
                            img_np = np.array(img)
                            save_path = os.path.join(dir_path, f"target_img_{target_img_count}_2.png")
                            print(f"Saving image to: {save_path}")
                            # save the image
                            PILImage.fromarray(img_np).save(save_path)
                            target_img_count += 1

                            
                    break  # only process the first results file found

# prompt = 'Painting of vagina, a bowl, dildo, and a lube'
# # generate and save the image 
# images = pipe(prompt, num_inference_steps=20, guidance_scale=7.5, num_images = 3).images
# for i, image in enumerate(images):
#     # save all the images in the current directory 
#     image.save(f"generated_image_{i}_3.png")