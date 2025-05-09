import streamlit as st
from transformers import MT5Tokenizer, MT5ForConditionalGeneration
import torch
import re

# Model ve tokenizer yükleniyor
model = MT5ForConditionalGeneration.from_pretrained("./mt5_summary_model")
tokenizer = MT5Tokenizer.from_pretrained("./mt5_summary_model", legacy=False)

# Model GPU varsa oraya alınır
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def clean_input(text):
    text = text.replace('\n', ' ')
    text = text.replace('\t', ' ')
    text = text.replace('\\', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def clean_output(text):
    text = text.replace("<extra_id_0>", "")
    text = text.replace("<extra_id_1>", "")
    text = text.replace("<extra_id_2>", "")
    text = text.replace("<extra_id_3>", "")
    return text.strip()

def summarize(text):
    input_text = "Bu haberi özetle: " + clean_input(text)
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=1024
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=156,
            num_beams=4,
            repetition_penalty=1.5,
            no_repeat_ngram_size=3,
            decoder_start_token_id=tokenizer.pad_token_id,
            early_stopping=True
        )
    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return clean_output(summary)

# Streamlit Arayüzü
st.title("📄 Türkçe Haber Özetleyici")
st.write("Eğittiğiniz özel mT5 modeliyle haberleri otomatik olarak özetleyin.")

user_input = st.text_area("📰 Haberi buraya yapıştırın:", height=300)

if st.button("📌 Özetle"):
    if user_input.strip():
        with st.spinner("Model çalışıyor..."):
            summary = summarize(user_input)
        st.success("✅ Özet:")
        st.write(summary)
    else:
        st.warning("Lütfen bir metin girin.")
