import os
import whisper
import streamlit as st
from gtts import gTTS
from jiwer import wer
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Load models
whisper_model = whisper.load_model("base")
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
translation_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
translator = pipeline("translation", model=translation_model, tokenizer=tokenizer, src_lang="eng_Latn", tgt_lang="tha_Thai")

# Paths
input_folder = "english_audio"
output_folder = "OUTPUT"
reference_file = "reference.txt"
os.makedirs(output_folder, exist_ok=True)

# Load ground truth
with open(reference_file, "r", encoding="utf-8") as f:
    references = dict(line.strip().split(" ", 1) for line in f.readlines())

st.title("🎙️ English to Thai Speech Translator with Accuracy Metrics")

for file in os.listdir(input_folder):
    if file.endswith(".mp3"):
        path = os.path.join(input_folder, file)

        # Transcribe
        result = whisper_model.transcribe(path)
        english_text = result["text"].strip()

        # Translate
        translated = translator(english_text)
        thai_text = translated[0]["translation_text"]

        # Generate Thai audio
        thai_audio_path = os.path.join(output_folder, file.replace(".mp3", "_thai.mp3"))
        if not os.path.exists(thai_audio_path):  # Skip regeneration
            tts = gTTS(text=thai_text, lang='th')
            tts.save(thai_audio_path)

        # Get ground truth and WER
        ref_text = references.get(file, "")
        transcription_wer = wer(ref_text, english_text) if ref_text else "N/A"

        # Show in Streamlit
        st.subheader(f"📄 File: {file}")
        st.audio(thai_audio_path)
        st.markdown(f"**📝 Transcribed English:** {english_text}")
        st.markdown(f"**🌏 Translated Thai:** {thai_text}")
        st.markdown(f"**✅ Accuracy (WER):** {transcription_wer if transcription_wer == 'N/A' else round(transcription_wer, 3)}")
        st.markdown("---")
