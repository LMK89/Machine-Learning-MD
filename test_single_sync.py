from deep_translator import GoogleTranslator
import os

translator = GoogleTranslator(source='en', target='vi')

def trans_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    paragraphs = text.split('\n\n')
    translated = []
    in_code = False
    for p in paragraphs:
        if p.startswith('```') and not in_code:
            in_code = True
            translated.append(p)
            continue
        if p.endswith('```') and in_code:
            in_code = False
            translated.append(p)
            continue
        if in_code or p.strip() == '':
            translated.append(p)
            continue
        try:
            translated.append(translator.translate(p))
        except:
            translated.append(p)

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(translated))

files = []
for root, _, fs in os.walk("LoTrinhThucChien"):
    for f in fs:
        if f.endswith(".md") and f != "plan.md":
            files.append(os.path.join(root, f))

# Only translate the first 10 sequentially to see if it works
for f in files[:10]:
    print("Translating:", f)
    trans_file(f)
print("Done translating 10 files.")
