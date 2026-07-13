from deep_translator import GoogleTranslator
import time

filepath = "LoTrinhThucChien/01_NenTang_Python_Toan/001_Probability_Distributions_Python_Cheat_Sheet.md"
with open(filepath, 'r', encoding='utf-8') as f:
    text = f.read()

translator = GoogleTranslator(source='auto', target='vi')

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

with open(filepath, 'w', encoding='utf-8') as f:
    f.write('\n\n'.join(translated))

print("Translated 1 file")
