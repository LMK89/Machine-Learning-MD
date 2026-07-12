from deep_translator import GoogleTranslator

translator = GoogleTranslator(source='en', target='vi')
try:
    print(translator.translate("Hello world"))
except Exception as e:
    print(f"Error: {e}")
