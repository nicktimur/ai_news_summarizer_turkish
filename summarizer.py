import json
import re
from datasets import Dataset
from transformers import MT5Tokenizer, MT5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq, EarlyStoppingCallback
import torch


# 1. Veriyi Yükle
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f.readlines()]

# 2. Dataset Hazırlığı
def create_dataset(data_path):
    raw_data = load_jsonl(data_path)
    return Dataset.from_list([
        {"text": item["text"], "summary": item["summary"]}
        for item in raw_data if item["text"].strip() and item["summary"].strip()
    ])

# 3. Tokenizer ve Model
model_name = "google/mt5-base"
tokenizer = MT5Tokenizer.from_pretrained(model_name, legacy=False)
model = MT5ForConditionalGeneration.from_pretrained(model_name)

def clean_text(text):
    text = text.replace('\n', ' ')
    text = text.replace('\t', ' ')
    text = text.replace('\\', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# 4. Ön işleme
def preprocess(example):
    input_text = "Bu haberi özetle: " + clean_text(example["text"])
    target_text = clean_text(example["summary"])

    input_encoding = tokenizer(
        input_text,
        max_length=1024,
        truncation=True,
        padding="max_length"
    )
    target_encoding = tokenizer(
        target_text,
        max_length=128,
        truncation=True,
        padding="max_length"
    )

    labels = [
        token if token != tokenizer.pad_token_id and token < tokenizer.vocab_size else -100
        for token in target_encoding["input_ids"]
    ]

    return {
        "input_ids": input_encoding["input_ids"],
        "attention_mask": input_encoding["attention_mask"],
        "labels": labels
    }

# 5. Dataset yükle ve işle
dataset = create_dataset("data/labeled_data.jsonl")
tokenized_dataset = dataset.map(
    preprocess,
    batched=False,
    remove_columns=["text", "summary"]
)

# 6. Eğitim ayarları
training_args = TrainingArguments(
    output_dir="./mt5_summary_model",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=6,
    learning_rate=2e-5,
    save_total_limit=2,
    logging_steps=5,
    save_steps=100,
    fp16=False,
    eval_strategy="steps",
    eval_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    remove_unused_columns=False
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=-100,
    return_tensors="pt",
    decoder_start_token_id=tokenizer.pad_token_id
)


# 7. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# 8. Manuel örnekle loss ölçümü
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Eğitimi başlatmak istersen:
    trainer.train()
    trainer.save_model("./mt5_summary_model")
    tokenizer.save_pretrained("./mt5_summary_model")
