import os
import requests
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from sconf import Config
from peft import LoraConfig, get_peft_model

model_path = 'microsoft/Phi-4-multimodal-instruct'
device = torch.device('cuda:0')


user_prompt = '<|user|>'
assistant_prompt = '<|assistant|>'
prompt_suffix = '<|end|>'



# Vision model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map=device,
    dtype=torch.bfloat16,
    attn_implementation='flash_attention_2',
)



config = Config('config_lora/llm_lora_config.yaml')
lora_config = LoraConfig(
                    r=config.r,
                    lora_alpha=config.lora_alpha,
                    lora_dropout=config.lora_dropout,
                    bias=config.bias if config.bias != 'none' else 'none',
                    task_type=config.task_type,
                    target_modules=config.target_modules if config.target_modules else None,
                )
model = get_peft_model(model, config)
print()


# processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
# generation_config = GenerationConfig.from_pretrained(model_path, 'generation_config.json')


# all_param = 0
# for _, param in model.named_parameters():
#     all_param += param.numel()

# print(all_param)

# # Prompt settings
# instruction = 'Find all the food and drinks in the image. Then describe the details of the food and drinks. And extract the location of the food in a normalized form between 0 and 1.'
# instruction = """
# Find all the food and drinks in the image. Then describe the details of the food and drinks. And extract the location of the food between 0 and 1 in the format [center_x, center_y, width, height]. Then, make this information as JSON format.
# Format example:
# {"apple": [0.12, 0.28, 0.35, 0.46], "banana": [0.39, 0.44, 0.28, 0.66], "water": [0.52, 0.63, 0.25, 0.47]}
# """
# prompt = f'{user_prompt}<|image_1|>{instruction}{prompt_suffix}{assistant_prompt}'
# image = Image.open('images/321883_202408030640.jpg')
# print(prompt)
# inputs = processor(text=prompt, images=image, return_tensors='pt').to(device)
# generate_ids = model.generate(
#     **inputs,
#     max_new_tokens=1000,
#     generation_config=generation_config,
# )
# generate_ids = generate_ids[:, inputs['input_ids'].shape[1] :]
# response = processor.batch_decode(
#     generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
# )[0]
# print(f'>>> Response\n{response}')


# # import matplotlib.pyplot as plt
# # import matplotlib.patches as patches

# # def draw_bounding_boxes_with_matplotlib(image_path, annotations):
# #     # Load the image
# #     image = plt.imread(image_path)
# #     height, width, _ = image.shape

# #     # Create a figure and axis
# #     fig, ax = plt.subplots(figsize=(20, 20))
# #     ax.imshow(image)

# #     for label, coords in annotations.items():
# #         for coord in coords:
# #             center_x, center_y, box_width, box_height = coord

# #             # Convert to top-left corner coordinates
# #             x1 = (center_x - box_width / 2) * width
# #             y1 = (center_y - box_height / 2) * height

# #             # Create a rectangle
# #             rect = patches.Rectangle((x1, y1), box_width * width, box_height * height, 
# #                                       linewidth=2, edgecolor='green', facecolor='none')
# #             ax.add_patch(rect)
# #             ax.text(x1, y1 - 5, label, color='green', fontsize=10, weight='bold')

# #     plt.axis('off')  # Turn off the axis
# #     plt.savefig('output.jpg', bbox_inches='tight')

# # # Example usage
# # annotations = {
# #     "rice": [
# #         [0.22, 0.55, 0.5, 0.87]
# #     ],
# #     "omelet": [
# #         [0.34, 0.38, 0.7, 0.6]
# #     ],
# #     "chopsticks": [
# #         [0.7, 0.45, 0.14, 0.55]
# #     ]
# # }

# # image_path = 'images/321883_202408030640.jpg'
# # draw_bounding_boxes_with_matplotlib(image_path, annotations)