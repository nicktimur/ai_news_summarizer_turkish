from playwright.sync_api import sync_playwright
import json
import os

new_data = []

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)  # ilk test için görünür
    page = browser.new_page()

    print("AA ana sayfa yükleniyor...")
    page.goto("https://www.aa.com.tr/tr/saglik", timeout=60000)
    page.wait_for_timeout(3000)

    article_links = set()
    max_clicks = 50
    click_count = 0
    unchanged_rounds = 0

    print("➕ Devam butonuna tıklanarak haberler yükleniyor...")

    while click_count < max_clicks:
        # Yeni bağlantıları topla
        hrefs = page.eval_on_selector_all("a", "els => els.map(e => e.getAttribute('href'))")
        new_links = {
            link for link in hrefs
            if link and link.startswith("https://www.aa.com.tr/tr/saglik/")
        }

        before = len(article_links)
        article_links.update(new_links)
        after = len(article_links)

        print(f"🔗 Toplam haber bağlantısı: {after}")

        # Devam butonuna tıkla (eğer varsa)
        try:
            devam_button = page.query_selector("a.button-daha")
            if devam_button:
                devam_button.click()
                click_count += 1
                page.wait_for_timeout(2500)
            else:
                print("❌ Devam butonu artık görünmüyor.")
                break
        except Exception as e:
            print(f"× Devam tıklamada hata: {e}")
            break

        # Aynı içerik tekrar yükleniyorsa sonlandır
        if after == before:
            unchanged_rounds += 1
        else:
            unchanged_rounds = 0

        if unchanged_rounds >= 3:
            print("🛑 Yeni içerik gelmiyor. Durduruluyor.")
            break

    print(f"\n✅ Toplam {len(article_links)} haber bağlantısı bulundu.\n")

    for link in list(article_links):
        try:
            page.goto(link)
            page.wait_for_timeout(2000)
            paragraphs = page.eval_on_selector_all(
                "div.detay-icerik p.selectionShareable",
                "els => els.map(e => e.innerText)"
            )
            text = " ".join(paragraphs).strip()
            if len(text) > 150:
                new_data.append({ "url": link, "text": text })
                print(f"✓ Eklendi: {link}")
        except Exception as e:
            print(f"× Hata (makale): {link} -> {e}")

    browser.close()

# 🔄 JSON'a ekle ve kaydet
data_path = "data/sample_news.json"
os.makedirs("data", exist_ok=True)

existing_data = []
if os.path.exists(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        existing_data = json.load(f)

existing_urls = set(item["url"] for item in existing_data)

for item in new_data:
    if item["url"] not in existing_urls and len(item["text"]) > 150:
        existing_data.append(item)
        existing_urls.add(item["url"])
        print(f"✓ Eklendi: {item['url']}")

with open(data_path, "w", encoding="utf-8") as f:
    json.dump(existing_data, f, ensure_ascii=False, indent=2)

print(f"\n📝 Toplam {len(existing_data)} haber başarıyla kaydedildi. Dosya: {data_path}")
