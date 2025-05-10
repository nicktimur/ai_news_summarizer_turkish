import gradio as gr
import torch
import re
from transformers import MT5ForConditionalGeneration, MT5Tokenizer

# Model yükleme (yalnızca bir kez)
model_path = "./mt5_summary_model"  # doğru klasör
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MT5ForConditionalGeneration.from_pretrained(model_path).to(device)
model.eval()
tokenizer = MT5Tokenizer.from_pretrained(model_path, legacy=False)

# bad words
bad_words_ids = [tokenizer.encode(f"<extra_id_{i}>", add_special_tokens=False) for i in range(100) if tokenizer.encode(f"<extra_id_{i}>", add_special_tokens=False)]

def clean_input(text):
    return re.sub(r'\s+', ' ', text.replace('\n', ' ').replace('\t', ' ').replace('\\', ' ')).strip()

def clean_output(text):
    return re.sub(r'<extra_id_\d+>', '', text).strip()

def summarize(text, method="beam"):
    input_text = "Özetle: " + clean_input(text)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=1024).to(device)

    with torch.no_grad():
        if method == "beam":
            outputs = model.generate(
                **inputs,
                max_length=156,
                num_beams=6,
                repetition_penalty=2.0,
                no_repeat_ngram_size=4,
                length_penalty=1.0,
                early_stopping=True,
                bad_words_ids=bad_words_ids,
                do_sample=False
            )
        else:
            outputs = model.generate(
                **inputs,
                max_length=256,
                do_sample=True,
                top_k=30,
                top_p=0.92,
                repetition_penalty=2.0,
                no_repeat_ngram_size=4,
                length_penalty=1.0,
                early_stopping=True,
                bad_words_ids=bad_words_ids,
                num_return_sequences=1
            )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return clean_output(summary)

# Gradio UI
iface = gr.Interface(
    fn=lambda text, method: summarize(text, method),
    inputs=[
        gr.Textbox(lines=10, placeholder="Buraya haberi yapıştırın", label="Haber Metni"),
        gr.Radio(choices=["beam", "sampling"], value="beam", label="Özetleme Yöntemi")
    ],
    outputs=gr.Textbox(label="Özet"),
    title="📄 Türkçe Haber Özetleyici",
    description="Eğitilmiş mT5 modeli ile Türkçe haber metinlerini özetleyin."
)

iface.launch()
