from faster_whisper import WhisperModel


# Load model once - using CPU for stable performance
# Once CUDA is properly installed, change to device="cuda" for GPU acceleration
model = WhisperModel("base", device="cpu", compute_type="float32")


def transcribe_audio(audio_path: str):
    segments, _ = model.transcribe(audio_path)

    transcript = ""
    for segment in segments:
        transcript += segment.text + " "

    return transcript.strip()