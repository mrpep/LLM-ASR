import gradio as gr
import torch
import torchaudio
from llmasr.tasks import load_model

# Load model
state = {}
model = load_model(state, 'models_to_evaluate/qwen1.5B-mhubert-catmlp-originalLR-addTedLium-weightedaverage-fixnograd/epoch-5-step-1848.ckpt') #COMPLETAR

def transcribe(audio):
    # Load the audio file
    x = {'filename': audio,
        'transcription': ''}
    for p in state['model'].input_processors:
        x = p(x)
    xin = state['model'].collate_fn([x])
    xin = {k: v.to(state['model'].device) if isinstance(v, torch.Tensor) else v for k,v in xin.items()}
    xin = {k: v.to(state['model'].dtype) if v.dtype not in [torch.int64, torch.int32, torch.int16] else v for k,v in xin.items()}

    out = state['model'].generate(xin, tokenizer=state['tokenizer'])

    return out

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