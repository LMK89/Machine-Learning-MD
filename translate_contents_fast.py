import os
import concurrent.futures
from deep_translator import GoogleTranslator

def translate_markdown(text):
    paragraphs = text.split('\n\n')
    translated_paragraphs = []
    in_code_block = False

    translator = GoogleTranslator(source='en', target='vi')

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
            if len(p) > 4000:
                chunks = [p[i:i+4000] for i in range(0, len(p), 4000)]
                t_chunks = [translator.translate(chunk) for chunk in chunks]
                translated_paragraphs.append(''.join(t_chunks))
            else:
                translated_paragraphs.append(translator.translate(p))
        except Exception as e:
            translated_paragraphs.append(p)

    return '\n\n'.join(translated_paragraphs)

def process_file(filepath):
    print(f"Translating: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()

    translated = translate_markdown(content)

    with open(filepath, 'w', encoding='utf-8') as file:
        file.write(translated)

files_to_process = []
for root, _, files in os.walk("LoTrinhThucChien"):
    for f in files:
        if f.endswith(".md") and f != "plan.md":
            files_to_process.append(os.path.join(root, f))

with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    executor.map(process_file, files_to_process)

print("Translation completed.")
