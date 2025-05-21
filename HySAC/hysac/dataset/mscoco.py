import json
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class MSCOCO(Dataset):
    def __init__(self, annotation_path, tokenizer_name='bert-base-uncased', max_length=30):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

        with open(annotation_path, 'r') as f:
            data = json.load(f)

        # Just extract all captions
        self.captions = [ann['caption'] for ann in data['annotations']][:5000]

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        caption = self.captions[idx]
        tokenized = self.tokenizer(caption,
                                   padding='max_length',
                                   truncation=True,
                                   max_length=self.max_length,
                                   return_tensors='pt')

        return {
            'caption': caption,
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0)
        }
    def get_all_prompt_and_categories(self):
        prompts = [el for el in self.captions]
        category = ['benign' for el in self.captions]
        return prompts, category

if __name__ == "__main__":
    dataset = MSCOCO('./annotations/captions_train2014.json')
    print(dataset[0][0])
