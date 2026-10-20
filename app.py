from runtime import ModelRunner, PROJECT_ROOT

import gc
import os
import re
from threading import Lock

import gradio as gr
import torch
from transformers import MT5ForConditionalGeneration, MT5Tokenizer


MODELS = {
    "nicktimur/mt5-base-turkish-news-summarizer": {
        "label": "712 haber ile eğitilen model",
        "local_dirs": ("mt5_summary_model-ilk", "mt5_summary_model"),
    },
    "nicktimur/mt5-base-turkish-news-summarizer-10k": {
        "label": "10.000 haber ile eğitilen model",
        "local_dirs": ("mt5_parquet_summary_model",),
    },
}

DEFAULT_MODEL = "nicktimur/mt5-base-turkish-news-summarizer-10k"

# NEWS_MODEL kullanılıyorsa ayrı bir seçenek olarak eklenir.
custom_model = os.environ.get("NEWS_MODEL")
if custom_model:
    MODELS.setdefault(
        custom_model,
        {
            "label": "Özel model (NEWS_MODEL)",
            "local_dirs": (),
        },
    )
    DEFAULT_MODEL = custom_model


def resolve_model_source(model_id):
    if model_id not in MODELS:
        raise ValueError("Listeden geçerli bir model seçin.")

    # Seçilen modele ait yerel kayıt varsa onu kullan.
    for folder in MODELS[model_id]["local_dirs"]:
        candidate = PROJECT_ROOT / folder
        if (candidate / "config.json").is_file():
            return str(candidate)

    # Özel model için klasör yolu desteği.
    expanded = os.path.expanduser(model_id)

    if os.path.isdir(expanded):
        return expanded

    project_candidate = PROJECT_ROOT / expanded
    if project_candidate.is_dir():
        return str(project_candidate)

    # Yerel kayıt yoksa Hugging Face modelini kullan.
    return model_id


def clean_input(text):
    text = text.replace("\n", " ").replace("\t", " ").replace("\\", " ")
    return re.sub(r"\s+", " ", text).strip()


def clean_output(text):
    return re.sub(r"<extra_id_\d+>", "", text).strip()


class ModelManager:
    def __init__(self):
        self._lock = Lock()
        self.model_id = None
        self.runner = None
        self.tokenizer = None
        self.bad_words_ids = []

    def _release(self):
        device = self.runner.device.type if self.runner is not None else None

        self.runner = None
        self.tokenizer = None
        self.bad_words_ids = []
        self.model_id = None

        gc.collect()

        if device == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()

    def _activate(self, model_id):
        # Aktif model tekrar seçildiyse yeniden yükleme.
        if model_id == self.model_id and self.runner is not None:
            return

        source = resolve_model_source(model_id)

        # Aynı anda iki büyük model bellekte tutulmaz.
        self._release()

        print(f"Model yükleniyor: {source}", flush=True)

        try:
            tokenizer = MT5Tokenizer.from_pretrained(
                source,
                legacy=False,
            )

            runner = ModelRunner(
                MT5ForConditionalGeneration.from_pretrained(source)
            )

            bad_words_ids = []

            for index in range(100):
                tokens = tokenizer.encode(
                    f"<extra_id_{index}>",
                    add_special_tokens=False,
                )
                if tokens:
                    bad_words_ids.append(tokens)

        except Exception:
            self._release()
            raise

        self.tokenizer = tokenizer
        self.runner = runner
        self.bad_words_ids = bad_words_ids
        self.model_id = model_id

    def _status(self):
        label = MODELS[self.model_id]["label"]
        return f"Hazır: {label} · İşlem birimi: {self.runner.device}"

    def activate(self, model_id):
        with self._lock:
            self._activate(model_id)
            return self._status()

    def summarize(self, text, method, model_id):
        if not text or not clean_input(text):
            raise ValueError("Özetlemek için bir haber metni girin.")

        if method not in ("beam", "sampling"):
            raise ValueError("Geçerli bir özetleme yöntemi seçin.")

        # Model değiştirme ve özetleme aynı anda yapılmaz.
        with self._lock:
            self._activate(model_id)

            inputs = self.tokenizer(
                "Özetle: " + clean_input(text),
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=1024,
            )

            if method == "beam":
                outputs = self.runner.generate(
                    inputs,
                    max_length=156,
                    num_beams=6,
                    repetition_penalty=2.0,
                    no_repeat_ngram_size=4,
                    length_penalty=0.5,
                    early_stopping=True,
                    bad_words_ids=self.bad_words_ids,
                    do_sample=False,
                )
            else:
                outputs = self.runner.generate(
                    inputs,
                    max_length=256,
                    do_sample=True,
                    top_k=30,
                    top_p=0.92,
                    repetition_penalty=2.0,
                    no_repeat_ngram_size=4,
                    length_penalty=1.0,
                    early_stopping=True,
                    bad_words_ids=self.bad_words_ids,
                    num_return_sequences=1,
                )

            summary = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True,
            )

            return clean_output(summary), self._status()


manager = ModelManager()


def summarize(text, method="beam", model_id=DEFAULT_MODEL):
    try:
        return manager.summarize(text, method, model_id)
    except Exception as exc:
        print(f"Özetleme hatası: {exc}", flush=True)
        raise gr.Error(f"Özet üretilemedi: {exc}") from exc


def switch_model(model_id):
    try:
        status = manager.activate(model_id)
        # Önceki modele ait özeti temizle.
        return "", status
    except Exception as exc:
        print(f"Model yükleme hatası: {exc}", flush=True)
        return (
            "",
            f"Model yüklenemedi: {exc}. "
            "Tekrar seçebilir veya Özetle'ye basabilirsiniz.",
        )


model_selector = gr.Dropdown(
    choices=[
        (info["label"], model_id)
        for model_id, info in MODELS.items()
    ],
    value=DEFAULT_MODEL,
    label="Özetleme modeli",
    interactive=True,
)

summary_output = gr.Textbox(label="Özet")

status_output = gr.Textbox(
    label="Model durumu",
    interactive=False,
    value="Model seçtiğinizde veya Özetle'ye bastığınızda yüklenecek.",
)

iface = gr.Interface(
    fn=summarize,
    inputs=[
        gr.Textbox(
            lines=10,
            placeholder="Buraya haberi yapıştırın",
            label="Haber Metni",
        ),
        gr.Radio(
            choices=["beam", "sampling"],
            value="beam",
            label="Özetleme Yöntemi",
        ),
        model_selector,
    ],
    outputs=[summary_output, status_output],
    title="📄 Türkçe Haber Özetleyici",
    description=(
        "712 veya 10.000 haber ile eğitilen modeli seçerek "
        "aynı haberi karşılaştırın. "
        "Bir modelin ilk indirilmesi zaman alabilir."
    ),
    submit_btn="Özetle",
)

with iface:
    model_selector.change(
        fn=switch_model,
        inputs=model_selector,
        outputs=[summary_output, status_output],
    )

if __name__ == "__main__":
    iface.queue().launch()