# 📄 Türkçe Haber Özetleyici

Bu proje, Türkçe haber metinlerini özetlemek için Google'ın mT5 modelini kullanan bir uygulamadır. Proje, haber metinlerini beam search veya sampling yöntemleriyle özetler ve Gradio tabanlı bir kullanıcı arayüzü sunar. Ayrıca, veri temizleme, manuel etiketleme ve model eğitimi gibi özellikler içerir.

## 🚀 Özellikler

- **Haber Özetleme**: Türkçe haber metinlerini beam search veya sampling yöntemleriyle özetler.
- **Gradio Arayüzü**: Kullanıcı dostu bir arayüz ile haber metinlerini kolayca özetleyebilirsiniz.
- **Manuel Etiketleme**: Haber metinlerini manuel olarak özetlemek için Streamlit tabanlı bir etiketleme aracı.
- **Veri Temizleme**: Haber metinlerini temizlemek ve tekrar eden içerikleri filtrelemek için araçlar.
- **Model Eğitimi**: mT5 modelini Türkçe haber özetleme için eğitmek üzere bir eğitim pipeline'ı içerir.

---

## 📊 Eğitim Günlüğü (Training Logs)

| Epoch | Train Loss | Eval Loss | Grad Norm | Learning Rate |
|-------|------------|-----------|-----------|----------------|
| 0.12  | 15.98      |           | 41460.79  | 1.955e-05      |
| 0.25  | 14.87      |           | 7548.44   | 1.905e-05      |
| 0.38  | 14.46      |           | 6171.39   | 1.855e-05      |
| 0.50  | 12.98      | 7.74      | 3173.07   | 1.805e-05      |
| 0.62  | 11.94      |           | 3397.85   | 1.755e-05      |
| 0.75  | 11.33      |           | 6715.34   | 1.705e-05      |
| 0.88  | 11.37      |           | 4861.55   | 1.655e-05      |
| 1.00  | 10.50      | 5.37      | 12531.24  | 1.605e-05      |
| 1.12  | 9.78       |           | 1363.31   | 1.555e-05      |
| 1.25  | 9.33       |           | 3557.01   | 1.505e-05      |
| 1.38  | 8.17       |           | 448.00    | 1.455e-05      |
| 1.50  | 7.58       | 3.38      | 845.93    | 1.405e-05      |
| 1.62  | 7.05       |           | 1770.12   | 1.355e-05      |
| 1.75  | 6.29       |           | 38.35     | 1.305e-05      |
| 1.88  | 5.68       |           | 139.72    | 1.255e-05      |
| 2.00  | 4.99       | 2.40      | 67.03     | 1.205e-05      |
| 2.12  | 4.60       |           | 73.81     | 1.155e-05      |
| 2.25  | 4.45       |           | 14.30     | 1.105e-05      |
| 2.38  | 3.87       |           | 18.73     | 1.055e-05      |
| 2.50  | 3.86       | 2.21      | 132.57    | 1.005e-05      |
| 2.62  | 3.48       |           | 6.23      | 9.55e-06       |
| 2.75  | 3.43       |           | 5.57      | 9.05e-06       |
| 2.88  | 3.27       |           | 6.03      | 8.55e-06       |
| 3.00  | 3.24       | 2.10      | 5.54      | 8.05e-06       |

- Bu eğitim yaklaşık 4,5 saat sürmüştür.
---

## 🧠 Model

- Model: `google/mt5-base`
- Dataset: Türkçe haber ve özet çiftlerinden oluşmaktadır.
- Eğitim süreci, her 0.5 epoch'ta bir değerlendirme (`eval`) içerir.
- Optimizasyon: `AdamW`, lineer learning rate scheduler, `EarlyStopping` ile birlikte.

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
www.timurkarakas.com.tr

---

## Lisans 📜

Bu proje MIT Lisansı ile lisanslanmıştır. Daha fazla bilgi için `LICENSE` dosyasına göz atabilirsiniz.
