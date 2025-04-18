
from googletrans import Translator

def translate_text(text, src_lang, target_lang):
    translator = Translator()
    result = translator.translate(text, src=src_lang, dest=target_lang)
    return result.text


if __name__ == "__main__":
    print("Simple Language Translator")
    source_lang = input("Enter source language (e.g., 'en' for English): ")
    target_lang = input("Enter target language (e.g., 'hi' for Hindi): ")
    text_to_translate = input("Enter text to translate: ")

    translated_text = translate_text(text_to_translate, source_lang, target_lang)
    print(f"Translated text: {translated_text}")
