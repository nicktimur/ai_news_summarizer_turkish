# 📄 Türkçe Haber Özetleyici

Bu proje, Türkçe haber metinlerini özetlemek için Google'ın mT5 modelini kullanan bir uygulamadır. Proje, haber metinlerini beam search veya sampling yöntemleriyle özetler ve Gradio tabanlı bir kullanıcı arayüzü sunar. Ayrıca, veri temizleme, manuel etiketleme ve model eğitimi gibi özellikler içerir.

## 🚀 Özellikler

- **Haber Özetleme**: Türkçe haber metinlerini beam search veya sampling yöntemleriyle özetler.
- **Gradio Arayüzü**: Kullanıcı dostu bir arayüz ile haber metinlerini kolayca özetleyebilirsiniz.
- **Manuel Etiketleme**: Haber metinlerini manuel olarak özetlemek için Streamlit tabanlı bir etiketleme aracı.
- **Veri Temizleme**: Haber metinlerini temizlemek ve tekrar eden içerikleri filtrelemek için araçlar.
- **Model Eğitimi**: mT5 modelini Türkçe haber özetleme için eğitmek üzere bir eğitim pipeline'ı içerir.

---

## 📂 Proje Yapısı

```plaintext
.
├── app.py                     # Gradio tabanlı haber özetleme uygulaması
├── model_trainer.py           # mT5 modelini eğitmek için kullanılan kod
├── data/                      # Veri işleme ve temizleme için dosyalar
│   ├── clean_duplicate_json.py
│   ├── clean_json_data.py
│   ├── labeler_app.py         # Manuel etiketleme arayüzü
│   ├── sample_news_clean.json # Temizlenmiş haber verisi
│   ├── labeled_data.jsonl     # Etiketlenmiş haber verisi
│   └── ...
├── mt5_summary_model/         # Eğitilmiş mT5 modeli
│   ├── config.json
│   ├── pytorch_model.bin
│   └── ...
├── requirements.txt           # Gerekli Python kütüphaneleri
└── README.md                  # Proje açıklamaları
```

---

## 🛠️ Kurulum

1. **Gerekli Kütüphaneleri Yükleyin**  
   Proje için gerekli Python kütüphanelerini yüklemek için aşağıdaki komutu çalıştırın:

   ```bash
   pip install -r requirements.txt
   ```

2. **Model Dosyalarını İndirin**  
   Eğitilmiş mT5 modelini `mt5_summary_model/` klasörüne yerleştirin. Eğer model yoksa, `model_trainer.py` dosyasını kullanarak eğitebilirsiniz.

3. **Gradio Uygulamasını Çalıştırın**  
   Haber özetleme uygulamasını başlatmak için aşağıdaki komutu çalıştırın:

   ```bash
   python app.py
   ```

---

## 📊 Model Eğitimi

Eğer modeli sıfırdan eğitmek isterseniz:

0. **Manuel Etiketleme Aracını Çalıştırın(Kendi verinizi çekerseniz.)**  
   Haber metinlerini manuel olarak özetlemek için Streamlit tabanlı aracı çalıştırabilirsiniz:

   ```bash
   streamlit run data/labeler_app.py
   ```

1. **Etiketlenmiş Veriyi Hazırlayın**  
   `data/labeled_data.jsonl` dosyasını oluşturun. Bu dosya, haber metinlerini ve özetlerini içermelidir.

2. **Modeli Eğitin**  
   `model_trainer.py` dosyasını çalıştırarak modeli eğitebilirsiniz:

   ```bash
   python model_trainer.py
   ```

3. **Eğitilmiş Modeli Kaydedin**  
   Model eğitimi tamamlandıktan sonra, model `mt5_summary_model/` klasörüne kaydedilecektir.

---

## 📋 Kullanım

### Gradio Arayüzü
- Haber metnini kutuya yapıştırın.
- Özetleme yöntemini seçin (Beam Search veya Sampling).
- "Submit" butonuna tıklayarak özetinizi alın.

### Manuel Etiketleme
- Haber metinlerini okuyarak özetlerini girin.
- Kaydet butonuna tıklayarak etiketlenmiş veriyi oluşturun.

---

## 📦 Gereksinimler

- Python 3.9 veya üzeri
- Gerekli kütüphaneler (`requirements.txt` içinde listelenmiştir)

---

## 📚 Kullanılan Teknolojiler

- **[Transformers](https://huggingface.co/transformers/)**: mT5 modeli için.
- **[Gradio](https://gradio.app/)**: Kullanıcı arayüzü için.
- **[Streamlit](https://streamlit.io/)**: Manuel etiketleme aracı için.
- **[Datasets](https://huggingface.co/docs/datasets/)**: Veri işleme için.

---

## 🤝 Katkıda Bulunma

Katkıda bulunmak isterseniz, lütfen bir pull request gönderin veya bir issue açın.

---

## 📧 İletişim

Herhangi bir sorunuz veya öneriniz varsa, lütfen benimle iletişime geçin.
timurkarakas.com.tr

---

