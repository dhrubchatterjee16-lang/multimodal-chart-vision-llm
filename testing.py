from PIL import Image

FastVisionModel.for_inference(model)

# Open image
image_input = Image.open("sample_chart.png").convert("RGB")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image_input},
            {"type": "text", "text": "Analyze this chart image and extract all visual data points into strict JSON."}
        ]
    }
]

prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
inputs = tokenizer(image_input, prompt, add_special_tokens=False, return_tensors="pt").to("cuda")

print("⚡ Running Visual Inference on Image...")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.1)

response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

print("="*60)
print("🎉 EXTRACTED VISUAL DATA FROM IMAGE:")
print("="*60)
print(response)
