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
    text = re.sub(r'<extra_id_\d+>, ', '', text).strip()
    text = re.sub(r'<extra_id_\d+>', '', text).strip()
    return text

# Özetleme fonksiyonu
def summarize(text):
    input_text = "Özetle: " + clean_input(text)
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
            max_length=256,
            min_length=50,
            num_beams=6,                
            repetition_penalty=2.0,     
            no_repeat_ngram_size=4,     
            length_penalty=1.0,
            early_stopping=True,
            do_sample=False        
        )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Özet: {summary}")
    return clean_output(summary)

def summarize_sampling(text):
    input_text = "Özetle: " + clean_input(text)
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
            max_length=256,
            do_sample=True,              # sampling aktif
            top_k=30,
            top_p=0.92,
            repetition_penalty=2.0,       # 🔼 Daha yüksek ceza, daha az kopya
            no_repeat_ngram_size=4,       # 🔼 4 kelimelik tekrarları engelle
            length_penalty=1.0,           # 🔁 Cümle uzunluğunu cezalandırmaz
            early_stopping=True,
            num_return_sequences=1,       # Tek üretim (çoklu üretimle kalite seçimi yapılabilir)
        )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Özet: {summary}")
    return clean_output(summary)

# Streamlit arayüzü
st.title("📄 Türkçe Haber Özetleyici")
st.write("Eğittiğiniz özel mT5 modeliyle haberleri otomatik olarak özetleyin.")

user_input = st.text_area("📰 Haberi buraya yapıştırın:", height=300)

if st.button("📌 Özetle"):
    if user_input.strip():
        with st.spinner("Model çalışıyor..."):
            print(user_input)
            summary = summarize(user_input)
        st.success("✅ Özet:")
        st.write(summary)
    else:
        st.warning("Lütfen bir metin girin.")

if st.button("📌 Doğal Özetle"):
    if user_input.strip():
        with st.spinner("Model çalışıyor..."):
            print(user_input)
            summary = summarize_sampling(user_input)
        st.success("✅ Özet:")
        st.write(summary)
    else:
        st.warning("Lütfen bir metin girin.")
