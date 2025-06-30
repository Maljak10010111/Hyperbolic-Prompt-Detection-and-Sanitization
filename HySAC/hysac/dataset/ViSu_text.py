from torch.utils.data import Dataset
from datasets import load_dataset
import os


class ViSuPrompts(Dataset):
    def __init__(
        self,
        cache_dir,
        split="train",
    ):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.data_path = os.path.join(self.cache_dir, "aimagelab___vi_su-text")
        

        data = load_dataset(cache_dir=self.cache_dir, path='/mnt/ssd1/mary/Diffusion-Models-Embedding-Space-Defense/.cache/aimagelab___vi_su-text/default/0.0.0/9afabb85b5570fa883b7caa0561d8c8d71d84dcd')
        self.data = {
            "train": data["train"],
            "validation": data["validation"],
            "test": data["test"],
        }
        self.split = split
        self.dataset = self.data[self.split]

    def __getitem__(self, index):
        element = self.dataset[index]
        safe_text = element["safe"]
        nsfw_text = element["nsfw"]

        return (safe_text, nsfw_text)

    def __len__(self):
        return len(self.dataset)

    def get_all_prompt_and_categories(self):
        prompts = []
        categories = []
        for el in self.dataset:
            prompts.append(el["safe"])
            categories.append("benign")
            prompts.append(el["nsfw"])
            categories.append("malicious")
        return prompts, categories


if __name__ == "__main__":
    # Example usage
    dataset = ViSuPrompts(
        cache_dir="/mnt/ssd1/mary/Diffusion-Models-Embedding-Space-Defense/.cache",
        split="train",
    )
    captions, nsfw_caption = dataset[0]
    print(f"Caption: {captions}, Category: {nsfw_caption}")
    print(f"Dataset length: {len(dataset)}")
