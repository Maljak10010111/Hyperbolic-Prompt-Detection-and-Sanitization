from datasets import load_dataset
import os
from dotenv import load_dotenv
import pandas as pd

# define the static class for the dataset I2P
load_dotenv()

class I2P:
    def __init__(self,  split, ):
        data = load_dataset("AIML-TUDA/i2p",
            cache_dir = os.getenv('TORCH_HOME')
        )
        self.data = data
        self.split = split
        self.name = 'i2p'
    def __len__(self):
        return len(self.data)

    def get_all_prompts(self, split = 'train'):
        prompt_list = [el for el in self.data[split].to_pandas()['prompt']]
        return prompt_list

    def get_all_prompt_and_categories(self, split = 'train'):
        prompt_list = [el for el in self.data[split].to_pandas()['prompt']]
        category_list = [el for el in self.data[split].to_pandas()['categories']]
       
        return prompt_list, category_list
    

if __name__ == "__main__":
    dataset = I2P( 'train')
    print(len(dataset))
    ds = dataset.data
    # 1. Basic Info
    # print("Dataset Structure:")
    # print(ds)

    # # 2. Feature Descriptions
    # print("\nFeatures and Types:")
    # print(ds['train'].features)

    # # 3. General Statistics
    # print("\nDataset Size:", len(ds['train']))
    # print("First 3 Samples:\n", ds['train'][:3])


    # # Convert to pandas for better statistics
    # df = df_train = ds['train'].to_pandas()
    # print("\nNumerical Summary:")
    # print(df.describe())

    # # 5. Distribution of Categorical Features (like categories)
    # print("\nCategory Distribution:")
    # print(df['categories'].value_counts())

    # # 6. Detect Missing Values
    # print("\nMissing Values per Column:")
    # print(df.isnull().sum())

    list_prompt, list_categ = dataset.get_all_prompt_and_categories()
    print(list_categ[4000])
    
    