import streamlit as st
import json
import os

# Dosya yolları
input_path = "data/sample_news_clean.json"
output_path = "data/labeled_data.jsonl"

# Veriyi yükle
with open(input_path, "r", encoding="utf-8") as f:
    news_data = json.load(f)

# Daha önce kaydedilmiş verileri URL'siz olarak kontrol etmek için text'e göre ayıklama
labeled_texts = set()
if os.path.exists(output_path):
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
                labeled_texts.add(entry.get("text"))
            except:
                continue

# Daha önce özetlenmemiş metinleri filtrele
unlabeled_data = [item for item in news_data if item.get("text") not in labeled_texts]

st.title("📝 Haber Özetleme Arayüzü")
st.markdown("Her haberin metnini okuyarak manuel olarak özetini girin ve kaydedin.")

if not unlabeled_data:
    st.success("Tüm haberler etiketlenmiş görünüyor. Harika!")
    st.stop()

# Sayfa durumu için session state kullan
if "index" not in st.session_state:
    st.session_state.index = 0

item = unlabeled_data[st.session_state.index]

st.subheader("Haber Metni")
st.text_area("", item["text"], height=300, disabled=True)

summary = st.text_area("🖊️ Bu haberin özeti:", "")

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    if st.button("💾 Kaydet"):
        if summary.strip():
            with open(output_path, "a", encoding="utf-8") as f:
                entry = {
                    "text": item["text"].strip(),
                    "summary": summary.strip()
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            st.success("Kaydedildi!")
            st.session_state.index += 1
            st.rerun()
        else:
            st.warning("Lütfen bir özet girin.")

with col2:
    if st.button("⏭️ Geç"):
        st.session_state.index += 1
        st.rerun()

with col3:
    st.markdown(f"**{st.session_state.index + 1} / {len(unlabeled_data)} haber gösteriliyor.**")
