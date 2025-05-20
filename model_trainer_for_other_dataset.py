import re
import torch
from datasets import load_dataset, Dataset
from transformers import (
    MT5ForConditionalGeneration,
    MT5Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
import evaluate

# ROUGE metriği
rouge = evaluate.load("rouge")

# Metin temizleme
def clean_text(text):
    text = text.replace('\n', ' ').replace('\t', ' ').replace('\\', ' ')
    return re.sub(r'\s+', ' ', text).strip()

# Veriyi işle
def preprocess(example):
    input_text = "Özetle: " + clean_text(example["text"])
    target_text = clean_text(example["summary"])

    model_inputs = tokenizer(
        input_text, max_length=1024, truncation=True, padding="max_length"
    )
    labels = tokenizer(
        target_text, max_length=128, truncation=True, padding="max_length"
    )["input_ids"]

    labels = [token if token != tokenizer.pad_token_id else -100 for token in labels]

    model_inputs["labels"] = labels
    return model_inputs

# ROUGE skorları
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    return rouge.compute(predictions=decoded_preds, references=decoded_labels)

# Model ve tokenizer
model_name = "google/mt5-base"
tokenizer = MT5Tokenizer.from_pretrained(model_name)
model = MT5ForConditionalGeneration.from_pretrained(model_name)

# .parquet dosyasını oku
dataset = load_dataset("parquet", data_files={"train": "data/news_summary.parquet"})

# 🔹 Sadece ilk 5000 örneği kullan
dataset = dataset["train"].select(range(10000)).train_test_split(test_size=0.1)

# Tokenize et
tokenized_train = dataset["train"].map(preprocess, remove_columns=["text", "summary"])
tokenized_test = dataset["test"].map(preprocess, remove_columns=["text", "summary"])


# Eğitim ayarları
training_args = TrainingArguments(
    output_dir="./mt5_parquet_summary_model",
    logging_dir="./logs",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    learning_rate=2e-5,
    logging_steps=10,
    save_steps=90,
    save_total_limit=2,
    eval_steps=90,
    eval_strategy="steps",
    save_strategy="steps",
    logging_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    remove_unused_columns=False,
    save_safetensors=False
)

# Trainer oluştur
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

# Eğitim
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(device)
    
    trainer.train()
    trainer.save_model("./mt5_parquet_summary_model")
    tokenizer.save_pretrained("./mt5_parquet_summary_model")
