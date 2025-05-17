from HySAC.hysac.dataset import i2p
from transformers import CLIPTokenizer
from HySAC.hysac.models import HySAC

from dotenv import load_dotenv

load_dotenv()
# load the I2P dataset for gettin
dataset = i2p.I2P('i2p', 'train')
all_prompts = dataset.get_all_prompts()
# print(len(all_prompts))
device = 'cuda:2'


model_id = "aimagelab/hysac"
clip_backbone = 'openai/clip-vit-large-patch14'
model = HySAC.from_pretrained(model_id, device=device).to(device)

tokenizer =  CLIPTokenizer.from_pretrained(clip_backbone)

first_text_prompt = all_prompts[0]
first_text_prompt_tokens = tokenizer(first_text_prompt, return_tensors='pt', padding='max_length', truncation=True)
# input ids 
first_text_input_ids = first_text_prompt_tokens['input_ids'].to(device)
first_text_input_ids = first_text_prompt_tokens['attention_mask'].to(device)

model.eval()

print('the text prompt is :', first_text_prompt)
print(first_text_prompt_tokens)

# text_encoding = model.encode_text(first_text_prompt_tokens, project = True)