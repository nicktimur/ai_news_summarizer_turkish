# MacBook kurulumu

Bu sürüm macOS'ta MPS/CPU, Windows'ta mevcut CUDA/CPU akışını kullanır.
Windows kurulumu ve özet üretme parametreleri korunmuştur. macOS bağımlılıkları
ayrı tutulur; eski requirements.txt içinde tekrarlanan sürümler Mac kurulumuna taşınmaz.

## Başlatma

Apple Silicon üzerinde native **arm64 Python 3.11 veya 3.12** kullanın.
Rosetta altında x86_64 Python kullanmayın. Proje klasöründe:

```bash
python3 -c "import platform; print(platform.machine())"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-macos.txt
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
python app.py
```

Terminalde yazılan yerel Gradio bağlantısını tarayıcıda açın. VS Code kullanıyorsanız
Python yorumlayıcısı olarak bu projenin `.venv/bin/python` dosyasını seçin.
MPS kullanılamıyorsa otomatik CPU seçilir. CPU'yu açıkça seçmek için:

```bash
NEWS_DEVICE=cpu python app.py
```

## Model seçimi ve ilk açılış

Mac'te önce proje içindeki `mt5_parquet_summary_model/` klasörü kullanılır.
Bu klasör yoksa README'de yayımladığınız
`nicktimur/mt5-base-turkish-news-summarizer-10k` modeli Hugging Face'ten indirilir.
İlk çalıştırma internet bağlantısı ve model boyutu kadar boş disk alanı gerektirir;
büyük ağırlıkların yüklenmesi zaman alabilir. Model ağırlıkları bu pakette yoktur.
Terminalde modelin kaynağı ve kullanılan işlem birimi yazılır.

Başka bir kayıt veya model klasörü kullanmak için:

```bash
NEWS_MODEL=nicktimur/mt5-base-turkish-news-summarizer python app.py
NEWS_MODEL=/tam/yol/model_klasoru python app.py
```

Windows'ta varsayılan model klasörü ve CUDA/CPU seçimi aynıdır; internetten modele
geçiş yalnızca macOS'ta varsayılan yerel klasör bulunmadığında yapılır.

## MPS hata yönetimi

`runtime.py`, torch yüklenmeden önce macOS için `PYTORCH_ENABLE_MPS_FALLBACK=1`
ayarlar. Bu, PyTorch'un desteklediği işlemler için CPU çekirdeğine geçmesini sağlar.
Özet üretirken kalan MPS/Metal destek hatalarında model CPU'ya taşınır ve aynı istek
bir kez yeniden denenir. Başka hatalar gizlenmez. Bu yeniden deneme, eğitim için
uygulanmaz; eğitim sırasında kalıcı MPS sorunu yaşanırsa CPU ile yeniden başlatın:

```bash
NEWS_DEVICE=cpu python model_trainer.py
```

## Eğitim ve değerlendirme

Her iki eğitim dosyası da Trainer'ın MPS desteğini kullanır. Mac'te batch=1,
gradient_accumulation_steps=8, değerlendirme batch=1 ve float32 kullanılır.
Windows'taki eğitim parametreleri korunur. Büyük mT5-base modelinin eğitim belleği,
özet üretiminden çok daha fazladır; MPS bulunması eğitimin belleğe sığacağını garanti etmez.

```bash
python model_trainer.py
python model_trainer_for_other_dataset.py
NEWS_MODEL=nicktimur/mt5-base-turkish-news-summarizer-10k python test.py
```

`test.py` varsayılan olarak eski `mt5_summary_model-ilk` kaydını değerlendirir.
Değerlendirilecek model bu klasörde yoksa `NEWS_MODEL` ile belirtin.

## Kontroller

Model indirmeden yapılan birim ve Windows davranış kontrolleri:

```bash
python -m unittest discover -s tests -v
```

Bu kontroller gerçek MPS performansını, model kalitesini veya Windows GPU çalışmasını
ölçmez. Bunlar hedef bilgisayarda gerçek modelle ayrıca denenmelidir.
