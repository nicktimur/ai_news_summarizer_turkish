# 📄 Türkçe Haber Özetleyici

**macOS / Apple Silicon kurulumu:** [MACOS.md](MACOS.md). Windows CUDA/CPU akışı korunmuştur; Mac için `requirements-macos.txt` kullanın.

Bu proje, Türkçe haber metinlerini özetlemek için Google'ın mT5 modelini kullanan bir uygulamadır. Proje, haber metinlerini beam search veya sampling yöntemleriyle özetler ve Gradio tabanlı bir kullanıcı arayüzü sunar. Ayrıca, veri temizleme, manuel etiketleme ve model eğitimi gibi özellikler içerir.

## 🚀 Özellikler

- **Haber Özetleme**: Türkçe haber metinlerini beam search veya sampling yöntemleriyle özetler.
- **Gradio Arayüzü**: Kullanıcı dostu bir arayüz ile haber metinlerini kolayca özetleyebilirsiniz.
- **Manuel Etiketleme**: Haber metinlerini manuel olarak özetlemek için Streamlit tabanlı bir etiketleme aracı.
- **Veri Temizleme**: Haber metinlerini temizlemek ve tekrar eden içerikleri filtrelemek için araçlar.
- **Model Eğitimi**: mT5 modelini Türkçe haber özetleme için eğitmek üzere bir eğitim pipeline'ı içerir.

---

## Kullanılan Veriler

Modelin eğitimi sırasında iki farklı veri seti kullanılmıştır:

### 1. Elle Etiketlenmiş Veri Seti (712 Haber)

İlk aşamada, 712 adet Türkçe haber verisi **elle toplanmış** ve her biri **ChatGPT yardımıyla tek tek özetlenerek** etiketlenmiştir. Bu veri seti, modelin Türkçe özetleme yapısını öğrenmesi için temel eğitim verisi olarak kullanılmıştır.

### 2. Yenigün Türkçe Haber Özetleme Verisi (10.000 Haber)

İkinci aşamada, [Yeniguno Turkish News Summary](https://huggingface.co/datasets/yeniguno/turkish-news-summary-onesentence) veri setinden 60.000 haber arasından **rastgele seçilen 10.000 örnek** kullanılmıştır. Bu veri seti, daha geniş kapsamlı haber çeşitliliği ile modelin genelleme yeteneğini artırmayı hedeflemiştir.

Bu iki aşamalı veri kullanımı sayesinde model hem özgün hem de çeşitli haber içerikleri üzerinde özetleme yeteneği kazanmıştır.

---

## Eğitim Ortamı ve Ayarları

### Donanım Bilgisi

Model eğitimi aşağıdaki donanım üzerinde gerçekleştirilmiştir:

- **İşlemci (CPU):** Intel Core i5-11400F (6 çekirdek / 12 iş parçacığı)
- **RAM:** 32 GB DDR4 3200 MHz
- **Ekran Kartı (GPU):** NVIDIA RTX 3070 (8 GB VRAM)

---

## Eğitim Ayarları (TrainingArguments)

Model, aşağıdaki `transformers.TrainingArguments` parametreleriyle eğitilmiştir:

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./mt5_summary_model",                # Eğitim sonrası modelin kaydedileceği dizin
    per_device_train_batch_size=4,                   # Her cihaz (GPU/CPU) için batch büyüklüğü
    gradient_accumulation_steps=2,                   # Toplamda 4x2 = 8 batch büyüklüğüne eşdeğer güncelleme için gradyan biriktirme
    num_train_epochs=5,                              # Verisetinin 5 kez modellenmesi
    learning_rate=2e-5,                              # Başlangıç öğrenme oranı
    save_total_limit=2,                              # Sadece en fazla 2 checkpoint saklanır
    logging_steps=10,                                # Her 10 adımda bir log kaydı alınır
    save_strategy="steps",                           # Belirli adımlarda checkpoint kaydetme stratejisi
    save_steps=40,                                   # Her 40 adımda bir model checkpoint kaydedilir
    fp16=False,                                      # FP16 (yarı hassasiyet) kullanımı devre dışı
    eval_strategy="steps",                           # Belirli adımlarda değerlendirme yapılır
    eval_steps=40,                                   # Her 40 adımda bir değerlendirme yapılır
    load_best_model_at_end=True,                     # Eğitim sonunda en iyi (en düşük loss'lu) model yüklenir
    metric_for_best_model="eval_loss",               # En iyi modelin belirlenmesinde kullanılan metrik
    greater_is_better=False,                         # Daha düşük `eval_loss` daha iyidir
    remove_unused_columns=False,                     # Kullanılmayan veri sütunları silinmez (özelleştirilmiş dataset'ler için)
    save_safetensors=False                           # `safetensors` formatı kullanılmaz
)
```

---

## 📊 Eğitim Günlüğü (Training Logs) / Benim oluşturduğum 712 veriyle

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

## 📈 ROUGE Skorları

| Metrik     | Değer   |
|------------|---------|
| ROUGE-1    | 0.2851  |
| ROUGE-2    | 0.1502  |
| ROUGE-L    | 0.1950  |
| ROUGE-Lsum | 0.1963  |

## 📄 İndirme Linki
[![Model: nicktimur/mt5-base-turkish-news-summarizer](https://img.shields.io/badge/HuggingFace-Model-yellow?logo=huggingface)](https://huggingface.co/nicktimur/mt5-base-turkish-news-summarizer)

---

## 📊 Eğitim Günlüğü (Training Logs) / Internetten aldığım 10k veri ile

| Epoch | Train Loss | Eval Loss | Grad Norm | Learning Rate |
|-------|------------|-----------|-----------|----------------|
| 0.08  | 9.83       | 5.13      | 2813.58   | 1.947e-05      |
| 0.16  | 5.15       | 2.41      | 147.49    | 1.893e-05      |
| 0.24  | 3.07       | 2.01      | 6.31      | 1.840e-05      |
| 0.32  | 2.93       | 1.84      | 8.44      | 1.787e-05      |
| 0.40  | 2.71       | 1.77      | 6.24      | 1.733e-05      |
| 0.48  | 2.65       | 1.73      | 4.63      | 1.680e-05      |
| 0.56  | 2.54       | 1.70      | 7.10      | 1.627e-05      |
| 0.64  | 2.43       | 1.66      | 3.68      | 1.573e-05      |
| 0.72  | 2.39       | 1.65      | 4.40      | 1.520e-05      |
| 0.80  | 2.38       | 1.63      | 4.76      | 1.467e-05      |
| 0.88  | 2.35       | 1.63      | 3.46      | 1.413e-05      |
| 0.96  | 2.32       | 1.60      | 4.00      | 1.360e-05      |
| 1.04  | 2.22       | 1.60      | 3.83      | 1.307e-05      |
| 1.12  | 2.09       | 1.59      | 5.84      | 1.253e-05      |
| 1.20  | 2.17       | 1.57      | 9.41      | 1.200e-05      |
| 1.28  | 2.31       | 1.57      | 4.33      | 1.147e-05      |
| 1.36  | 2.23       | 1.56      | 7.75      | 1.093e-05      |
| 1.44  | 2.23       | 1.55      | 3.53      | 1.040e-05      |
| 1.52  | 2.24       | 1.55      | 4.44      | 9.872e-06      |
| 1.60  | 2.07       | 1.54      | 4.57      | 9.339e-06      |
| 1.68  | 2.16       | 1.53      | 4.23      | 8.805e-06      |
| 1.76  | 2.05       | 1.53      | 7.17      | 8.272e-06      |
| 1.84  | 2.03       | 1.52      | 10.10     | 7.739e-06      |
| 1.92  | 2.08       | 1.51      | 9.33      | 7.205e-06      |
| 2.00  | 2.03       | 1.50      | 25.33     | 6.672e-06      |
| 2.08  | 2.14       | 1.50      | 4.55      | 6.139e-06      |
| 2.16  | 2.03       | 1.50      | 3.68      | 5.605e-06      |
| 2.24  | 2.08       | 1.51      | 5.41      | 5.072e-06      |

⏱ **Toplam Eğitim Süresi:** `306089 saniye` ≈ **85 saat 1 dakika**

## 📈 ROUGE Skorları (Değerlendirme Sonucu)

| Metik     | Değer   |
|-----------|---------|
| ROUGE-1   | 0.3873  |
| ROUGE-2   | 0.2389  |
| ROUGE-L   | 0.3305  |
| ROUGE-Lsum| 0.3309  |

## 📄 İndirme Linki
[![Model: nicktimur/mt5-base-turkish-news-summarizer-10k](https://img.shields.io/badge/HuggingFace-Model-yellow?logo=huggingface)](https://huggingface.co/nicktimur/mt5-base-turkish-news-summarizer-10k)

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
