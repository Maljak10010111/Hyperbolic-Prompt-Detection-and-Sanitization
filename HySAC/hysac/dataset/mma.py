import csv
from torch.utils.data import Dataset


class MMA(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (str): Path to the .csv file with CIV prompts.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.samples = []
        self.transform = transform

        with open(csv_file, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append(
                    {
                        "target_prompt": row["target_prompt"],
                        "adv_prompt": row["adv_prompt"],
                        "sanitized_adv_prompt": row["sanitized_adv_prompt"],
                        "success": row["success_against_sanitization_defense"].lower()
                        in ("true", "1", "yes"),
                        "clean_prompt": row["clean_prompt"],
                    }
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample
    def get_all_prompt_and_categories(self):
        prompts = []
        categories = []
        print('len all prompts', len(self.samples))
        for el in self.samples:
            prompts.append(el['target_prompt'])
            categories.append('malicious')
            prompts.append(el['sanitized_adv_prompt'])
            categories.append('benign')
        print('len prompts', len(prompts))
        print('len cat', len(categories))
        return prompts, categories

if __name__ == "__main__":
    dataset = MMA(
        csv_file="/mnt/ssd1/mary/Diffusion-Models-Embedding-Space-Defense/.cache/mma-diffusion-nsfw-adv-prompts.csv"
    )
