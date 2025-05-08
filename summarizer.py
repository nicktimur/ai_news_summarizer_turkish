# summarizer.py
# Türkçe Otomatik Özetleme için Fine-Tuning ve Inference Kodu

from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from datasets import load_dataset
import torch

# Model ve tokenizer yükle
model = MT5ForConditionalGeneration.from_pretrained("csebuetnlp/mT5_multilingual_XLSum")
tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")

# Cihaz ayarı
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Veri setini yükle (Türkçe)
dataset = load_dataset("csebuetnlp/xlsum", "turkish")
train_data = dataset['train']

# Eğitim örneklerini hazırlama
def preprocess(example):
    input_text = "generate summary: " + example["text"]
    target_text = example["summary"]
    inputs = tokenizer(input_text, max_length=512, truncation=True, padding="max_length", return_tensors="pt")
    targets = tokenizer(target_text, max_length=80, truncation=True, padding="max_length", return_tensors="pt")
    return {
        "input_ids": inputs.input_ids.squeeze(),
        "attention_mask": inputs.attention_mask.squeeze(),
        "labels": targets.input_ids.squeeze(),
    }

encoded_dataset = train_data.map(preprocess, remove_columns=train_data.column_names)

from torch.utils.data import DataLoader
train_loader = DataLoader(encoded_dataset, batch_size=2, shuffle=True)

# Eğitim döngüsü
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

model.train()
for epoch in range(1):  # Hızlı test için 1 epoch yeterli, artırabilirsin
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"Epoch {epoch+1} tamamlandı. Loss: {loss.item()}")

# Modeli kaydet
model.save_pretrained("./models/turkish_mt5_summarizer")
tokenizer.save_pretrained("./models/turkish_mt5_summarizer")

print("Eğitim tamamlandı ve model kaydedildi.")
