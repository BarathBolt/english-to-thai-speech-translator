from flask import Flask, request, jsonify, send_file
import os
import whisper
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from gtts import gTTS
import uuid
import logging
from pydub import AudioSegment

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static', static_url_path='/static')

# Ensure static folder exists
if not os.path.exists(app.static_folder):
    os.makedirs(app.static_folder)

# Load models (loaded once at startup)
whisper_model = whisper.load_model("base")
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
translator = pipeline("translation", model=model, tokenizer=tokenizer, src_lang="eng_Latn", tgt_lang="tha_Thai")

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/process_audio', methods=['POST'])
def process_audio():
    try:
        if 'file' not in request.files:
            logger.error("No file uploaded")
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        filename = file.filename.lower()
        logger.debug("Received file: %s", filename)

        # Determine if the file is recorded audio (WebM) or uploaded MP3
        is_recorded = filename.endswith('.webm')
        if not (is_recorded or filename.endswith('.mp3')):
            logger.error("Invalid file type: %s", filename)
            return jsonify({'error': 'Only MP3 or recorded audio (WebM) files are supported'}), 400

        # Check file size (max 10MB)
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        if file_size > 10 * 1024 * 1024:  # 10MB
            logger.error("File too large: %d bytes", file_size)
            return jsonify({'error': 'File too large (max 10MB)'}), 400
        file.seek(0)

        # Save the input file
        temp_input = os.path.join(app.static_folder, f"input_{uuid.uuid4()}{'.webm' if is_recorded else '.mp3'}")
        logger.debug("Saving input file to: %s", temp_input)
        file.save(temp_input)

        # Convert WebM to MP3 if recorded
        if is_recorded:
            try:
                temp_mp3 = os.path.join(app.static_folder, f"converted_{uuid.uuid4()}.mp3")
                logger.debug("Converting WebM to MP3: %s -> %s", temp_input, temp_mp3)
                audio = AudioSegment.from_file(temp_input, format="webm")
                audio.export(temp_mp3, format="mp3")
                os.remove(temp_input)
                temp_input = temp_mp3
                logger.debug("Conversion successful")
            except Exception as e:
                logger.error("Failed to convert WebM to MP3: %s", str(e))
                os.remove(temp_input)
                return jsonify({'error': 'Failed to convert recorded audio to MP3'}), 500

        # Step 1: Transcribe with Whisper
        logger.debug("Transcribing audio with Whisper")
        result = whisper_model.transcribe(temp_input)
        english_text = result["text"]
        logger.debug("Transcribed text: %s", english_text)

        # Step 2: Translate to Thai
        logger.debug("Translating to Thai")
        translated = translator(english_text)
        thai_text = translated[0]["translation_text"]
        logger.debug("Translated text: %s", thai_text)

        # Step 3: Generate Thai speech
        logger.debug("Generating Thai speech with gTTS")
        try:
            tts = gTTS(text=thai_text, lang='th')
        except Exception as e:
            logger.error("gTTS failed: %s", str(e))
            os.remove(temp_input)
            return jsonify({'error': 'Failed to generate speech'}), 500

        temp_output = os.path.join(app.static_folder, f"output_{uuid.uuid4()}.mp3")
        logger.debug("Saving output file to: %s", temp_output)
        tts.save(temp_output)

        # Validate MP3 file
        try:
            audio = AudioSegment.from_mp3(temp_output)
            logger.debug("MP3 file validated successfully. Duration: %d ms", audio.duration_seconds * 1000)
        except Exception as e:
            logger.error("Invalid MP3 file: %s", str(e))
            os.remove(temp_input)
            if os.path.exists(temp_output):
                os.remove(temp_output)
            return jsonify({'error': 'Failed to generate valid MP3 file'}), 500

        # Clean up input file
        os.remove(temp_input)
        logger.debug("Cleaned up input file: %s", temp_input)

        # Prepare response
        response = {
            'english_text': english_text,
            'thai_text': thai_text,
            'audio_url': f'/static/{os.path.basename(temp_output)}'
        }
        logger.debug("Response prepared: %s", response)

        return jsonify(response)

    except Exception as e:
        logger.error("Error in process_audio: %s", str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)