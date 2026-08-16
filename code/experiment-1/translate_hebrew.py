import csv
import re
from deep_translator import GoogleTranslator

INPUT_FILE = "C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 1/Experiment1_logs_cleaned_keepable_paired.csv"
OUTPUT_FILE = "C:/Users/alonr/OneDrive/Documents/LLM creativity/Experiment 1/Experiment1_logs_cleaned_keepable_paired_translated.csv"

HEBREW_PATTERN = re.compile(r'[א-ת]')

translator = GoogleTranslator(source="auto", target="en")

def translate_message(text: str) -> str:
    # Google Translate handles mixed Hebrew/English well with source="auto"
    return translator.translate(text)

def main():
    rows = []
    translated_count = 0

    with open(INPUT_FILE, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for i, row in enumerate(reader):
            msg = row.get("message", "")
            if HEBREW_PATTERN.search(msg):
                print(f"Translating row {i+2} (msg_id={row.get('message_id')})...")
                row["message"] = translate_message(msg)
                translated_count += 1
            rows.append(row)

    with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. Translated {translated_count} rows. Output: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
