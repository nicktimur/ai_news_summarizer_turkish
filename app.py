from transformers import MT5Tokenizer, MT5ForConditionalGeneration
import torch

# Model ve tokenizer yükleniyor
model = MT5ForConditionalGeneration.from_pretrained("./mt5_summary_model")
tokenizer = MT5Tokenizer.from_pretrained("./mt5_summary_model", legacy=False)

def summarize(text):
    input_text = "Özetlenecek haber: " + text.strip()
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=1024
    )
    model.eval()
    with torch.no_grad():
        output_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=156,
            num_beams=4,
            repetition_penalty=1.5,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
    summary = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    #summary = summary.replace("<extra_id_0>", "").strip()
    return summary


# Test
test_text = """
Ankara’da yaşayan 22 yaşındaki yazılım geliştiricisi Timur Karakaş, çocukluk hayali olan özel üretim Xbox 360 konsolunu bulmak için Türkiye’yi baştan sona dolaştı. 2005 yılında piyasaya sürülen ve günümüzde koleksiyon değeri taşıyan bu model, özellikle “Halo 3 Limited Edition” versiyonuyla büyük ilgi görüyor.

Timur, “Pandemiden sonra kendime küçük bir ödül vermek istedim. Eskiden oynadığım oyunları tekrar deneyimlemek istiyordum ama orijinal cihazı bulmak sanıldığından çok daha zordu.” diyerek başladığı yolculuğu İstanbul'dan Gaziantep’e kadar uzandı. Sahafları, ikinci el dükkanlarını ve forumları tek tek dolaşan Timur, sonunda Konya’da bir koleksiyoncu ile iletişime geçti.

Uzun pazarlıklar sonucunda nadir bulunan bu Xbox 360 modeli için 11.250 TL ödedi. Cihazın yanında gelen orijinal kutu, kablolar ve bir adet sınırlı sayıda üretilmiş oyun kolu da Timur’un yüzünü güldürdü. “Bunu sadece oyun oynamak için değil, anılarımı yaşatmak için aldım. Herkesin bir zaman kapsülü vardır, benimki bu cihaz.” dedi.

Sosyal medyada paylaştığı deneyimi kısa sürede binlerce kişiye ulaştı. Timur şimdi de benzer tutkulara sahip koleksiyoncular için küçük bir YouTube kanalı açmayı planlıyor."""
print(summarize(test_text))
