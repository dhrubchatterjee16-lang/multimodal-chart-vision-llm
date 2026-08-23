import torch
from unsloth import FastVisionModel

print("⌛ Loading Qwen2-VL 7B Vision Model in 4-bit mode...")
model, tokenizer = FastVisionModel.from_pretrained(
    model_name = "unsloth/Qwen2-VL-7B-Instruct-bnb-4bit",
    load_in_4bit = True,
    use_gradient_checkpointing = "unsloth",
)

# Attach LoRA adapters to both Vision Encoder & Language Decoder layers
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers = True,   # Fine-tune vision encoder
    finetune_language_layers = True, # Fine-tune language decoder
    r = 16,
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
)
print("✅ Multimodal Vision-Language Model ready for fine-tuning!")
