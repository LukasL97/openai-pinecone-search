
import os
from datasets import load_dataset

dataset = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)
dataset = dataset.take(1000)

output_directory = "data"
os.makedirs(output_directory, exist_ok=True)

for i, row in enumerate(dataset):
    title = row['title']
    text = row['text']
    file_path = os.path.join(output_directory, f"{title}.txt")
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(text)
