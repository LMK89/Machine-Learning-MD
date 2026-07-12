import os
from deep_translator import GoogleTranslator
import concurrent.futures

translator = GoogleTranslator(source='en', target='vi')

def translate_markdown(text):
    paragraphs = text.split('\n\n')
    translated_paragraphs = []
    in_code_block = False

    for p in paragraphs:
        if p.strip() == '':
            translated_paragraphs.append('')
            continue

        if p.startswith('```') and not in_code_block:
            in_code_block = True
            translated_paragraphs.append(p)
            continue
        elif p.endswith('```') and in_code_block:
            in_code_block = False
            translated_paragraphs.append(p)
            continue
        elif in_code_block:
            translated_paragraphs.append(p)
            continue

        if p.startswith('    ') or p.startswith('\t'):
            translated_paragraphs.append(p)
            continue

        try:
            if len(p) > 3000:
                chunks = [p[i:i+3000] for i in range(0, len(p), 3000)]
                t_chunks = [translator.translate(chunk) for chunk in chunks]
                translated_paragraphs.append(''.join(t_chunks))
            else:
                translated_paragraphs.append(translator.translate(p))
        except Exception as e:
            translated_paragraphs.append(p)

    return '\n\n'.join(translated_paragraphs)

def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()

    # Check if roughly English (if it has "Slide 1" or "The " a lot)
    if "Slide 1:" in content or " the " in content.lower():
        print(f"Translating: {filepath}")
        translated = translate_markdown(content)
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(translated)
    else:
        print(f"Skipping already translated: {filepath}")

files_to_process = []
for root, _, files in os.walk("LoTrinhThucChien"):
    for f in files:
        if f.endswith(".md") and f != "plan.md":
            files_to_process.append(os.path.join(root, f))

# Slow down to respect API
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    executor.map(process_file, files_to_process)

print("Translation pass 2 completed.")
