import json
import torch
from datasets import Dataset
from unsloth import FastVisionModel
from trl import SFTTrainer, SFTConfig

# 1. Ensure Model is Loaded in 4-bit (Run this if model isn't loaded)
try:
    model
except NameError:
    print("⌛ Loading Qwen2-VL 7B Vision Model...")
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name = "unsloth/Qwen2-VL-7B-Instruct-bnb-4bit",
        load_in_4bit = True,
        use_gradient_checkpointing = "unsloth",
    )
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers = True,
        finetune_language_layers = True,
        r = 16,
        lora_alpha = 16,
        bias = "none",
    )

# 2. Read and Normalize JSONL Data Structure
with open("vlm_dataset.jsonl", "r", encoding="utf-8") as f:
    raw_sample = json.loads(f.readline().strip())

# Clean message structure so Unsloth VLM Chat Template can parse it
image_path = raw_sample.get("image", "sample_chart.png")

# Ensure assistant's content is always a list of dictionaries for consistency
assistant_raw_content = raw_sample["messages"][-1]["content"]
if isinstance(assistant_raw_content, str):
    assistant_content_formatted = [{"type": "text", "text": assistant_raw_content}]
else:
    # This case should ideally not be hit if vlm_dataset.jsonl is always string
    # but handles potential future changes where it might already be a list.
    assistant_content_formatted = assistant_raw_content

normalized_messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": "Analyze this chart image and extract all visual data points into strict JSON."}
        ]
    },
    {
        "role": "assistant",
        "content": assistant_content_formatted # Use the consistently formatted content
    }
]

# 3. Create Dataset and Apply VLM Chat Template
vlm_dataset = Dataset.from_list([{"messages": normalized_messages}])

def apply_vlm_template(sample):
    text_prompt = tokenizer.apply_chat_template(
        sample["messages"],
        tokenize = False,
        add_generation_prompt = False
    )
    return {"text": text_prompt}

formatted_dataset = vlm_dataset.map(apply_vlm_template)

# 4. Configure SFTTrainer for VLM Fine-Tuning
training_args = SFTConfig(
    output_dir = "vlm_outputs",
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 4,
    max_steps = 30,
    learning_rate = 2e-4,
    fp16 = not torch.cuda.is_bf16_supported(),
    bf16 = torch.cuda.is_bf16_supported(),
    logging_steps = 5,
    optim = "adamw_8bit",
    dataset_text_field = "text",
    max_seq_length = 2048,
    remove_unused_columns = False,
)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = formatted_dataset,
    dataset_text_field = "text",
    max_seq_length = 2048,
    args = training_args,
)

# 5. Run Training
print("⚡ Starting Multimodal Vision Fine-Tuning...")
trainer.train()
print("🎉 VLM Fine-Tuning Complete!")

# 6. Save Adapter
model.save_pretrained("vlm_vision_adapter")
tokenizer.save_pretrained("vlm_vision_adapter")
print("💾 Saved VLM Vision Adapter to 'vlm_vision_adapter'!")
