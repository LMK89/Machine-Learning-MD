import os
import re
from collections import Counter

def get_keywords():
    keywords = Counter()
    for root, dirs, files in os.walk('.'):
        if '.git' in root or '.github' in root or 'LoTrinhThucChien' in root:
            continue
        for file in files:
            if file.endswith('.md'):
                # Extract words from filename
                name = file[:-3] # remove .md
                words = re.findall(r'\b[a-zA-Z]{3,}\b', name)
                for w in words:
                    keywords[w.lower()] += 1
    return keywords

if __name__ == '__main__':
    kws = get_keywords()
    for k, v in kws.most_common(50):
        print(f"{k}: {v}")
