# app.py
# Gradio arayüzü ile Türkçe özetleme

import gradio as gr
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
import torch

# Model ve tokenizer'ı yükle
model_path = "./models/turkish_mt5_summarizer"
model = MT5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = MT5Tokenizer.from_pretrained(model_path)

# Cihaz ayarı
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Özetleme fonksiyonu
def summarize(text):
    input_text = "generate summary: " + text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)
    summary_ids = model.generate(inputs["input_ids"], max_length=80, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Gradio arayüzü oluştur
interface = gr.Interface(fn=summarize, 
                         inputs=gr.Textbox(lines=10, placeholder="Uzun Türkçe metni buraya girin..."), 
                         outputs=gr.Textbox(label="Özet"),
                         title="Türkçe Otomatik Özetleme")

interface.launch()
