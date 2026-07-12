import os
import shutil
import re
from deep_translator import GoogleTranslator

# BLACKLIST: Outdated, purely historical, or overly theoretical terms
BLACKLIST = ["history", "evolution", "naive", "id3", "origin", "introduction to", "fundamentals", "what is", "primer", "explained", "concepts"]

# Phase mapping with HIGHLY practical keywords
PLAN = [
    (
        "LoTrinhThucChien/01_NenTang_Python_Toan",
        [("Python", "Python"), ("Python", "Code"), ("Statistics-and-Math", "Matrix"), ("Statistics-and-Math", "Derivative"), ("Python", "Practical")],
        50
    ),
    (
        "LoTrinhThucChien/02_PhanTichDuLieu_DataScience",
        [("Data-Science", "Pandas"), ("Data-Science", "Cleaning"), ("Data-Science", "NumPy"), ("Data-Visualization", "Matplotlib"), ("Data-Visualization", "Seaborn"), ("Data-Engineering", "SQL")],
        50
    ),
    (
        "LoTrinhThucChien/03_HocMay_MachineLearning",
        [("Machine-Learning", "Regression"), ("Machine-Learning", "XGBoost"), ("Machine-Learning", "Scikit"), ("Machine-Learning", "Gradient Descent"), ("Machine-Learning", "Pipeline")],
        50
    ),
    (
        "LoTrinhThucChien/04_HocSau_AI_ThucChien",
        [("Computer-Vision", "PyTorch"), ("Computer-Vision", "CNN"), ("NLP-and-Transformers", "NLP"), ("LLMs-and-GenAI", "Transformer"), ("LLMs-and-GenAI", "RAG"), ("LLMs-and-GenAI", "LLM")],
        50
    )
]

def is_practical(filename):
    lower_name = filename.lower()
    for bad in BLACKLIST:
        if bad in lower_name:
            return False
    return True

def find_files(base_dir, keyword, limit):
    found = []
    if not os.path.exists(base_dir):
        return found
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.endswith(".md") and keyword.lower() in f.lower() and is_practical(f):
                found.append(os.path.join(root, f))
                if len(found) >= limit:
                    return found
    return found

translator = GoogleTranslator(source='en', target='vi')

# Clear existing files to avoid conflicts, except plan.md and index.html
for d in os.listdir("LoTrinhThucChien"):
    path = os.path.join("LoTrinhThucChien", d)
    if os.path.isdir(path):
        for f in os.listdir(path):
            os.remove(os.path.join(path, f))

for dest_dir, keyword_sources, total_limit in PLAN:
    limit_per_kw = max(1, total_limit // len(keyword_sources))
    collected = []
    for src_dir, kw in keyword_sources:
        collected.extend(find_files(src_dir, kw, limit_per_kw))

    collected = list(set(collected))[:total_limit]

    # If still not enough, grab more randomly but STRICTLY practical
    if len(collected) < total_limit:
        for src_dir, _ in keyword_sources:
            if len(collected) >= total_limit:
                break
            for root, _, files in os.walk(src_dir):
                for f in files:
                    if len(collected) >= total_limit:
                        break
                    if f.endswith(".md") and f != "README.md" and is_practical(f):
                        path = os.path.join(root, f)
                        if path not in collected:
                            collected.append(path)

    for idx, filepath in enumerate(collected):
        filename = os.path.basename(filepath)
        name_no_ext = filename[:-3]

        try:
            translated_name = translator.translate(name_no_ext)
            translated_name = re.sub(r'[^\w\s-]', "", translated_name)
            new_filename = f"{idx+1:03d}_{translated_name}.md"
        except Exception as e:
            new_filename = f"{idx+1:03d}_{name_no_ext}.md"

        new_filename = new_filename.replace(" ", "_")

        dest_path = os.path.join(dest_dir, new_filename)
        shutil.copy2(filepath, dest_path)

print("Done copying and renaming 200 STRICTLY PRACTICAL files.")
