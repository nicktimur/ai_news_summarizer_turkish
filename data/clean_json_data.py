import json
import re

# 1. Temizleme fonksiyonu
def clean_text(text):
    # 1. "KAYNAK," ifadesini kaldır (ama "KAYNAK" kelimesini yalnız bırak)
    text = re.sub(r"\bKAYNAK,\s*", "", text, flags=re.IGNORECASE)

    # 2. Diğer kaynak isimlerini kaldır (GETTY IMAGES, NEWSMAKERS)
    text = re.sub(r"\b(GETTY IMAGES|NEWSMAKERS)\b[,/\-–— ]*", "", text, flags=re.IGNORECASE)

    # 3. BBC takip metinlerini sil
    text = re.sub(r"Gündemi BBC Türkçe.*?tıklayın\.", "", text, flags=re.DOTALL)

    # 4. "Haberin sonu" gibi kalıpları kaldır
    text = re.sub(r"Haberin sonu", "", text, flags=re.IGNORECASE)

    # 5. URL'leri kaldır
    text = re.sub(r"https?://\S+", "", text)

    # 6. Fazla boşlukları düzelt
    text = re.sub(r"\s+", " ", text)

    return text.strip()

# 2. JSON verisini oku
with open("data/sample_news.json", "r", encoding="utf-8") as file:
    news_data = json.load(file)

# 3. Her haberin text'ini temizle
cleaned_news = []

for item in news_data:
    cleaned_item = {
        "url": item.get("url", ""),
        "text": clean_text(item.get("text", ""))
    }
    cleaned_news.append(cleaned_item)

# 4. Temizlenmiş veriyi kaydet
with open("data/cleaned_news.json", "w", encoding="utf-8") as file:
    json.dump(cleaned_news, file, ensure_ascii=False, indent=4)

print("✅ Temizlenmiş veri cleaned_news.json dosyasına kaydedildi.")
