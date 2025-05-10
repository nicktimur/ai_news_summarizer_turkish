import streamlit as st
import torch
import re
from transformers import MT5ForConditionalGeneration, AutoTokenizer

# Model yolu ve cihaz ayarı
model_path = "./mt5_summary_model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model ve tokenizer yükle (AutoTokenizer ile uyumluluk sağlanır)
model = MT5ForConditionalGeneration.from_pretrained(model_path).to(device)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False)

# Temizleme fonksiyonları
def clean_input(text):
    text = re.sub(r'\s+', ' ', text.replace('\n', ' ').replace('\t', ' ').replace('\\', ' '))
    return text.strip()

def clean_output(text):
    return re.sub(r'<extra_id_\d+>', '', text).strip()

# Özetleme fonksiyonu
def summarize(text):
    input_text = "Haber özeti üret: " + clean_input(text)
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=1024
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=156,
            num_beams=4,
            repetition_penalty=1.5,
            no_repeat_ngram_size=3,
            early_stopping=True
        )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return clean_output(summary)

# Streamlit arayüzü
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
