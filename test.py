import json
import random
import re
from datasets import Dataset
from transformers import MT5Tokenizer, MT5ForConditionalGeneration
import torch
import evaluate
import matplotlib.pyplot as plt
import pandas as pd

# ROUGE metriğini yükle
rouge = evaluate.load("rouge")

# Model ve tokenizer yükle
model_path = "./mt5_summary_model-ilk"
#model_path = "./mt5_parquet_summary_model"
tokenizer = MT5Tokenizer.from_pretrained(model_path, legacy=False)
model = MT5ForConditionalGeneration.from_pretrained(model_path)
model.eval().to("cuda" if torch.cuda.is_available() else "cpu")

# Temizleme fonksiyonu
def clean_text(text):
    text = text.replace('\n', ' ')
    text = text.replace('\t', ' ')
    text = text.replace('\\', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Dataset yükle
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f.readlines()]
    
def load_parquet(path):
    df = pd.read_parquet(path)
    return df.to_dict(orient="records")

# Rastgele veri seç
def get_sample_subset(dataset, ratio=0.1):
    sample_size = int(len(dataset) * ratio)
    return random.sample(dataset, sample_size)

# Özet üretme
def generate_summary(text):
    input_text = "Özetle: " + clean_text(text)
    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512).to(model.device)

    summary_ids = model.generate(
        input_ids,
        max_length=128,
        num_beams=4,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Ana metrik hesaplama fonksiyonu
def evaluate_model(data_path):
    raw_data = load_jsonl(data_path)
    #raw_data = load_parquet("data/news_summary.parquet")
    subset = get_sample_subset(raw_data)
    print("Örnek veri boyutu:", len(subset))
    print("Test başlıyor...")

    predictions = []
    references = []

    for item in subset:
        text = item["text"]
        reference = clean_text(item["summary"])
        pred = generate_summary(text)

        references.append(reference)
        predictions.append(pred)

    result = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
    return result

# Görselleştirme
def plot_rouge_scores(scores):
    keys = ["rouge1", "rouge2", "rougeL"]
    values = [scores[k] for k in keys]

    plt.figure(figsize=(8, 5))
    plt.bar(keys, values)
    plt.title("ROUGE Skorları (%10 rastgele örnek)")
    plt.ylim(0, 1)
    plt.ylabel("Skor")
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    scores = evaluate_model("data/labeled_data.jsonl")
    print("ROUGE Scores:", scores)
    plot_rouge_scores(scores)
