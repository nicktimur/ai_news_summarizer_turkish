import json
import os

# Girdi/Çıktı dosyaları
input_path = "data/sample_news_clean.json"
output_path = "data/sample_news_cleaned.json"

# Veriyi yükle
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"🧾 Başlangıçta {len(data)} haber var.")

# Yalnızca text'e göre tekrarları filtrele
seen_texts = set()
cleaned_data = []

for item in data:
    text = item.get("text", "").strip()

    if not text or len(text) < 100:
        continue  # boş veya çok kısa içerikleri at

    if text not in seen_texts:
        cleaned_data.append(item)
        seen_texts.add(text)

print(f"✅ Temizlenmiş veri: {len(cleaned_data)} haber kaldı (sadece text bazlı).")

# Dosyaya yaz
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

print(f"📁 Kaydedildi: {output_path}")
