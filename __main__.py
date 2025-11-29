import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration


IMAGE_PATH = "/home/saltchicken/Pictures/test.png"
PROMPT = "Write a long descriptive caption for this image in a formal tone."
MODEL_NAME = "fancyfeast/llama-joycaption-beta-one-hf-llava"


# Load JoyCaption
# bfloat16 is the native dtype of the LLM used in JoyCaption (Llama 3.1)
# device_map=0 loads the model into the first GPU
processor = AutoProcessor.from_pretrained(MODEL_NAME)
llava_model = LlavaForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype="bfloat16", device_map=0)
llava_model.eval()

with torch.no_grad():
	# Load image
	image = Image.open(IMAGE_PATH)

	# Build the conversation
	convo = [
		{
			"role": "system",
			"content": "You are a helpful image captioner.",
		},
		{
			"role": "user",
			"content": PROMPT,
		},
	]

	# Format the conversation
	convo_string = processor.apply_chat_template(convo, tokenize = False, add_generation_prompt = True)
	assert isinstance(convo_string, str)

	# Process the inputs
	inputs = processor(text=[convo_string], images=[image], return_tensors="pt").to('cuda')
	inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)

	# Generate the captions
	generate_ids = llava_model.generate(
		**inputs,
		max_new_tokens=512,
		do_sample=True,
		suppress_tokens=None,
		use_cache=True,
		temperature=0.6,
		top_k=None,
		top_p=0.9,
	)[0]

	# Trim off the prompt
	generate_ids = generate_ids[inputs['input_ids'].shape[1]:]

	# Decode the caption
	caption = processor.tokenizer.decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
	caption = caption.strip()
	print(caption)
