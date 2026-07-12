import os
import re
from collections import Counter
import json

def get_keywords():
    keywords = Counter()
    for root, dirs, files in os.walk('.'):
        if '.git' in root or '.github' in root or 'LoTrinhThucChien' in root:
            continue
        for file in files:
            if file.endswith('.md'):
                name = file[:-3] # remove .md
                words = re.findall(r'\b[a-zA-Z]{4,}\b', name)
                for w in words:
                    keywords[w.lower()] += 1
    return keywords

kws = get_keywords()
top_100 = kws.most_common(100)

with open('LoTrinhThucChien/keywords_2000.txt', 'w', encoding='utf-8') as f:
    f.write("=== TOP 100 KEYWORDS TỪ 2000 FILES ===\n")
    for k, v in top_100:
        f.write(f"#{k}: {v} bài\n")

# Save JSON for HTML injection
with open('keywords_data.json', 'w') as f:
    json.dump([k for k, v in top_100[:30]], f)

print("Extracted keywords from 2000 files.")
