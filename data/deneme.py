from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()
    page.goto("https://www.bbc.com/turkce", timeout=60000)
    page.wait_for_timeout(10000)  # bekleme süresi artırıldı

    html = page.content()
    with open("bbc_turkce_dump.html", "w", encoding="utf-8") as f:
        f.write(html)

    browser.close()

print("Sayfa HTML içeriği bbc_turkce_dump.html dosyasına kaydedildi.")
