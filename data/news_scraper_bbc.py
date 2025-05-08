from playwright.sync_api import sync_playwright
import json
import os

os.makedirs("data", exist_ok=True)
data = []

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()

    print("BBC Türkçe ana sayfa yükleniyor...")
    page.goto("https://www.bbc.com/turkce", timeout=60000)
    page.wait_for_timeout(8000)  # JavaScript yüklenmesi için yeterli süre

    # 1. Tüm <a> etiketlerinden href'leri al
    hrefs = page.eval_on_selector_all(
        "a", "elements => elements.map(e => e.getAttribute('href'))"
    )

    # 2. /turkce/topics/ linklerini filtrele
    topic_links = list({
        "https://www.bbc.com" + href
        for href in hrefs
        if href and href.startswith("/turkce/topics/")
    })

    print(f"{len(topic_links)} topic bulundu.")

    # 3. Topics içinden makale linklerini topla
    article_links = set()
    for topic_url in topic_links:
        try:
            page.goto(topic_url, timeout=60000)
            page.wait_for_timeout(5000)
            links = page.eval_on_selector_all(
                "a", "elements => elements.map(e => e.getAttribute('href'))"
            )
            for href in links:
                if href and href.startswith("https://www.bbc.com/turkce/articles/"):
                    full_link = href
                    article_links.add(full_link)
            print(f"✓ Tarandı: {topic_url} - {len(article_links)} makale birikti.")

            # Erken durdurma: 1000 habere ulaştıysak dur
            if len(article_links) >= 1000:
                break
        except Exception as e:
            print(f"× Hata (topic): {topic_url} -> {e}")

    print(f"Toplam {len(article_links)} makale linki bulundu.")

    # 4. Her makaleyi ziyaret et ve metni topla
    for link in list(article_links)[:1000]:
        try:
            page.goto(link, timeout=60000)
            page.wait_for_timeout(3000)
            paragraphs = page.eval_on_selector_all(
                "main p", "elements => elements.map(e => e.innerText)"
            )
            text = " ".join(paragraphs).strip()
            if len(text) > 150:
                data.append({ "url": link, "text": text })
                print(f"✓ Eklendi: {link}")
        except Exception as e:
            print(f"× Hata (makale): {link} -> {e}")

    browser.close()

# 5. JSON'a kaydet
with open("data/sample_news.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"\nToplam {len(data)} haber başarıyla kaydedildi. Dosya: data/sample_news.json")
