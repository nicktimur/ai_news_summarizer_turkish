import json

def convert_to_text_format(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    with open(output_file, "w", encoding="utf-8") as f:
        for item in data:
            f.write(f"Summary: {item['text']}\n\n")

if __name__ == "__main__":
    convert_to_text_format("data/sample_news.json", "data/formatted_data.txt")
