from datasets import load_dataset
import os
import re
from html import unescape

DATA_DIR = "data"
MAX_RECORDS = 400


def clean_text(text: str) -> str:
    if not text:
        return ""

    text = unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    ds = load_dataset("jacob-hugging-face/job-descriptions", split="train")
    os.makedirs(DATA_DIR, exist_ok=True)

    for i, entry in enumerate(ds.select(range(MAX_RECORDS))):
        title = clean_text(entry.get("position_title", "Unknown Title"))
        company = clean_text(entry.get("company_name", "Unknown Company"))
        description = clean_text(entry.get("job_description", ""))

        filename = os.path.join(DATA_DIR, f"job_{i}.txt")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(
                f"Job Title: {title}\n"
                f"Company: {company}\n\n"
                f"Job Description:\n{description}"
            )

    print("Done! Cleaned data saved in 'data' folder.")


if __name__ == "__main__":
    main()