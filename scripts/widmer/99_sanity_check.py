from pathlib import Path

# File path
path = Path(r"C:\Users\ymw04\Dropbox\shifting_slant\scripts\newspapers\classification\econ\02_llm_labels_gemini_thinking.csv")

# Read raw content
raw = path.read_text(encoding="utf-8").strip()

# Insert newlines
fixed = raw.replace("article_id,label ", "article_id,label\n").replace(" ", "\n")

# Overwrite file
path.write_text(fixed, encoding="utf-8")

print("Fixed CSV with proper line breaks.")
