import os
import shutil
import re
from deep_translator import GoogleTranslator

# Mapping phase to source keywords
# List of target tuples: (destination_dir, list_of_keywords, number_of_files_to_pick)
PLAN = [
    (
        "LoTrinhThucChien/01_NenTang_Python_Toan",
        [("Python/Core-Language", "Python"), ("Python/Object-Oriented", "Class"), ("Statistics-and-Math/Linear-Algebra-and-Matrices", "Matrix"), ("Statistics-and-Math/Calculus-and-Analysis", "Derivative"), ("Statistics-and-Math/Probability-and-Distributions", "Probability")],
        20
    ),
    (
        "LoTrinhThucChien/02_PhanTichDuLieu_DataScience",
        [("Data-Science", "Pandas"), ("Data-Science", "Cleaning"), ("Data-Science", "NumPy"), ("Data-Visualization", "Matplotlib"), ("Data-Visualization", "Seaborn")],
        20
    ),
    (
        "LoTrinhThucChien/03_HocMay_MachineLearning",
        [("Machine-Learning/Regression", "Linear Regression"), ("Machine-Learning/Classification-and-Clustering", "Classification"), ("Machine-Learning/Classification-and-Clustering", "Clustering"), ("Deep-Learning/Backpropagation-and-Optimization", "Gradient Descent"), ("Machine-Learning", "Scikit")],
        20
    ),
    (
        "LoTrinhThucChien/04_HocSau_AI_ThucChien",
        [("Computer-Vision", "Neural Network"), ("Computer-Vision", "CNN"), ("NLP-and-Transformers", "NLP"), ("LLMs-and-GenAI", "Transformer"), ("LLMs-and-GenAI", "LLM")],
        20
    )
]

def find_files(base_dir, keyword, limit):
    found = []
    if not os.path.exists(base_dir):
        return found
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.endswith(".md") and keyword.lower() in f.lower():
                found.append(os.path.join(root, f))
                if len(found) >= limit:
                    return found

    if len(found) < limit:
        for root, _, files in os.walk(base_dir):
            for f in files:
                if f.endswith(".md") and f != "README.md":
                    path = os.path.join(root, f)
                    if path not in found:
                        found.append(path)
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

    collected = list(set(collected))[:total_limit] # Ensure uniqueness

    for idx, filepath in enumerate(collected):
        filename = os.path.basename(filepath)
        name_no_ext = filename[:-3]

        try:
            translated_name = translator.translate(name_no_ext)
            translated_name = re.sub(r'[\\/*?:"<>|]', "", translated_name) # Clean invalid chars
            new_filename = f"{idx+1:02d}_{translated_name}.md"
        except Exception as e:
            print(f"Failed to translate {name_no_ext}: {e}")
            new_filename = f"{idx+1:02d}_{name_no_ext}.md"

        dest_path = os.path.join(dest_dir, new_filename)
        shutil.copy2(filepath, dest_path)
        print(f"Copied: {filepath} -> {dest_path}")

print("Done copying and renaming 80 files.")
