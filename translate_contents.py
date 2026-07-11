import os
import time
from deep_translator import GoogleTranslator

translator = GoogleTranslator(source='en', target='vi')

def translate_markdown(text):
    # Split by lines or paragraphs to avoid Google Translate length limits (5000 chars)
    # A simple approach: split by double newline (paragraphs)
    paragraphs = text.split('\n\n')
    translated_paragraphs = []

    in_code_block = False

    for p in paragraphs:
        if p.strip() == '':
            translated_paragraphs.append('')
            continue

        # Don't translate code blocks entirely
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

        # Avoid translating pure code formatting if it starts with spaces/tabs
        if p.startswith('    ') or p.startswith('\t'):
            translated_paragraphs.append(p)
            continue

        try:
            # chunking logic if a single paragraph is too large
            if len(p) > 4500:
                chunks = [p[i:i+4500] for i in range(0, len(p), 4500)]
                t_chunks = [translator.translate(chunk) for chunk in chunks]
                translated_paragraphs.append(''.join(t_chunks))
            else:
                translated_paragraphs.append(translator.translate(p))
            time.sleep(0.5) # respect rate limits
        except Exception as e:
            print(f"Error translating chunk: {e}")
            translated_paragraphs.append(p) # fallback to original

    return '\n\n'.join(translated_paragraphs)


base_dir = "LoTrinhThucChien"

for root, _, files in os.walk(base_dir):
    for f in files:
        if f.endswith(".md") and f != "plan.md":
            filepath = os.path.join(root, f)
            print(f"Translating content of: {filepath}")

            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()

            translated_content = translate_markdown(content)

            with open(filepath, 'w', encoding='utf-8') as file:
                file.write(translated_content)

print("Done translating contents.")
