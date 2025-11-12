from flask import Flask, render_template, request, send_file, jsonify
from transformers import pipeline
from dotenv import load_dotenv
import os, uuid, datetime
import numpy as np
import scipy.io.wavfile

# ------------------ Flask Setup ------------------
app = Flask(__name__, template_folder="templates", static_folder="static")
os.makedirs("generated_audio", exist_ok=True)

# Load environment variables
load_dotenv()

# ------------------ AI Model Setup ------------------
print("🎧 Loading AI models...")
synthesizer = None
try:
    synthesizer = pipeline("text-to-audio", model="facebook/musicgen-small")
    print("✅ MusicGen model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading MusicGen: {e}")

# Store music generation history
history = []

# ------------------ Routes ------------------

@app.route("/")
def home():
    """Landing page with intro and navigation"""
    return render_template("home.html")

@app.route("/composer")
def composer():
    """AI Music generation interface"""
    return render_template("index.html")

@app.route("/history")
def music_history():
    """Page to view previously generated tracks"""
    return render_template("history.html", history=history)

@app.route("/generate", methods=["POST"])
def generate_music():
    """Generate AI-based music"""
    if synthesizer is None:
        return jsonify({"error": "Music generation model not available"}), 500

    try:
        data = request.get_json()
        prompt = data.get("prompt", "")
        genre = data.get("genre", "Any")
        mood = data.get("mood", "Calm")

        if not prompt:
            return jsonify({"error": "Please enter a theme or prompt"}), 400

        # Build final descriptive prompt
        full_prompt = f"Create a {genre} style track with a {mood} mood. Theme: {prompt}"
        print(f"🎵 Generating: {full_prompt}")

        # Generate audio
        result = synthesizer(full_prompt, forward_params={"do_sample": True, "max_length": 256})
        audio = np.clip(result["audio"], -1.0, 1.0)
        audio_int16 = (audio * 32767).astype(np.int16)

        # Save generated file
        uid = str(uuid.uuid4())[:8]
        path = f"generated_audio/music_{uid}.wav"
        scipy.io.wavfile.write(path, rate=result["sampling_rate"], data=audio_int16)

        # Add to history
        history.append({
            "id": uid,
            "prompt": prompt,
            "genre": genre,
            "mood": mood,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "url": f"/download/{uid}"
        })

        return jsonify({"url": f"/download/{uid}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/download/<uid>")
def download(uid):
    """Download generated music"""
    path = f"generated_audio/music_{uid}.wav"
    if not os.path.exists(path):
        return "File not found", 404
    return send_file(path, as_attachment=True, download_name=f"music_{uid}.wav")

# ------------------ Auto-Launch & Run ------------------
if __name__ == "__main__":
    import webbrowser
    webbrowser.open("http://127.0.0.1:5000")
    app.run(debug=True)
