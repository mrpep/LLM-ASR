import gradio as gr
import torch
import torchaudio

# Load model
model = None #COMPLETAR

def transcribe(audio):
    # Load the audio file
    waveform, sample_rate = torchaudio.load(audio)

    return f"{waveform.shape} / {sample_rate}" # Mock output

# Create the Gradio interface
interface = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(sources=["microphone", "upload"], type="filepath", label="Record or Upload an Audio File"), # Record audio in demo
    outputs=gr.Textbox(label="Transcription"),
    title="ASR Demo",
    description="Record or upload an audio file and get the transcription."
)

# Launch the interface
if __name__ == "__main__":
    interface.launch()