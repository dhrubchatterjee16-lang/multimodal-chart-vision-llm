import json
import matplotlib.pyplot as plt

# 1. Generate Synthetic Bar Chart PNG Image
categories = ['Reaction A', 'Reaction B', 'Reaction C', 'Reaction D']
yields = [65.4, 88.2, 72.1, 94.5]

plt.figure(figsize=(6, 4))
bars = plt.bar(categories, yields, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
plt.title('Chemical Yield Comparison Across Reactions (%)', fontsize=11, fontweight='bold')
plt.xlabel('Reaction Mixture', fontsize=9)
plt.ylabel('Yield (%)', fontsize=9)
plt.ylim(0, 100)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f'{yval}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig("sample_chart.png", dpi=150)
plt.close()
print("✅ Generated synthetic test image 'sample_chart.png'!")

# 2. Format VLM Training Dataset
vlm_sample = {
    "image": "sample_chart.png",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "sample_chart.png"},
                {"type": "text", "text": "Analyze this chart image and extract all visual data points into strict JSON."}
            ]
        },
        {
            "role": "assistant",
            "content": "```json\n{\n  \"title\": \"Chemical Yield Comparison Across Reactions (%)\",\n  \"chart_type\": \"BAR_CHART\",\n  \"extracted_data_points\": [\n    {\"label\": \"Reaction A\", \"value\": 65.4, \"unit\": \"%\"},\n    {\"label\": \"Reaction B\", \"value\": 88.2, \"unit\": \"%\"},\n    {\"label\": \"Reaction C\", \"value\": 72.1, \"unit\": \"%\"},\n    {\"label\": \"Reaction D\", \"value\": 94.5, \"unit\": \"%\"}\n  ],\n  \"visual_summary\": \"Bar chart showing yield for 4 reaction mixtures. Reaction D has the highest yield at 94.5%.\"\n}\n```"
        }
    ]
}

with open("vlm_dataset.jsonl", "w", encoding="utf-8") as f:
    f.write(json.dumps(vlm_sample) + "\n")

print("✅ Formatted VLM Multimodal dataset 'vlm_dataset.jsonl'!")
