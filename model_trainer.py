import json
import re
from datasets import Dataset
from transformers import MT5Tokenizer, MT5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq, EarlyStoppingCallback
import torch
import evaluate

rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # -100 olanları temizle (padding yerine)
    decoded_labels = [
        label.replace(tokenizer.pad_token, "") for label in decoded_labels
    ]

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    
    # Yalnızca rougeL ve rouge2 döndürmek istersen filtrele:
    return {
        "rouge1": result["rouge1"],
        "rouge2": result["rouge2"],
        "rougeL": result["rougeL"]
    }


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
    input_text = "Özetle: " + clean_text(example["text"])
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
dataset = create_dataset("data/labeled_data.jsonl").train_test_split(test_size=0.1)
train_dataset = dataset["train"]
test_dataset = dataset["test"]

tokenized_train_dataset = train_dataset.map(
    preprocess,
    batched=False,
    remove_columns=["text", "summary"]
)

tokenized_test_dataset = test_dataset.map(
    preprocess,
    batched=False,
    remove_columns=["text", "summary"]
)

# 6. Eğitim ayarları
training_args = TrainingArguments(
    output_dir="./mt5_summary_model",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    learning_rate=2e-5,
    label_smoothing_factor=0.1,
    save_total_limit=2,
    logging_steps=10,
    save_strategy="steps",
    save_steps=40,
    fp16=False,
    tf32=True,
    eval_strategy="steps",
    eval_steps=40,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    remove_unused_columns=False,
    save_safetensors=False
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=-100,
    return_tensors="pt"
)

# 7. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    #compute_metrics=compute_metrics, #Vram yetmedi
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# 8. Manuel örnekle loss ölçümü
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Eğitimi başlatmak istersen:
    trainer.train()
    #trainer.train(resume_from_checkpoint="./mt5_summary_model/checkpoint-5")
    trainer.save_model("./mt5_summary_model")
    tokenizer.save_pretrained("./mt5_summary_model")