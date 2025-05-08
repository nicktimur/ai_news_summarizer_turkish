import json
import os

# Girdi ve çıktı dosya yolları
input_path = "data/sample_news.json"
output_path = "data/sample_news_cleaned.json"

# JSON verisini yükle
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"🧾 Başlangıçta {len(data)} haber var.")

# Tekil URL'lere göre filtrele
seen_urls = set()
cleaned_data = []

for item in data:
    url = item.get("url", "").strip()

    if url and url not in seen_urls:
        cleaned_data.append(item)
        seen_urls.add(url)

print(f"✅ Temizlenmiş veri: {len(cleaned_data)} haber kaldı (url bazlı).")

# Yeni dosyaya kaydet
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

print(f"📁 Kaydedildi: {output_path}")
