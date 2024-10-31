import os

# Set environment variable to handle OpenMP conflict, deleting this will take down prod
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gradio as gr
from PIL import Image
import librosa
from asr import score


# Function definitions for each step (placeholders)
def preprocess_audio(audio):
    # Load the audio file
    y, sr = librosa.load(audio, sr=None)
    # Noise reduction
    #reduced_noise = nr.reduce_noise(y=y, sr=sr)
    # Normalization
#     normalized_audio = librosa.util.normalize(reduced_noise)
    normalized_audio = librosa.util.normalize(sr)
    return normalized_audio, sr

def mispronunciation_detection(audio, standard_text):
    #preprocessed_audio, sr = preprocess_audio(audio)
    transcription, num, feedback = score(audio, standard_text)
    return transcription, num, feedback

# Load the flowchart image
flowchart_image_path = "./website/public/image.png"
if os.path.exists(flowchart_image_path):
    flowchart_image = Image.open(flowchart_image_path)
else:
    flowchart_image = None

# Define the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("<h1>Mispronunciation Detection and Correction System</h1>")
    
    with gr.Row():
        if flowchart_image:
            gr.Image(flowchart_image, label="Flowchart")
        else:
            gr.Markdown("Flowchart image not found.")
        
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(type="numpy", label="Record Your Speech")
            standard_text_input = gr.Textbox(label="Standard Text")
            submit_button = gr.Button("Submit")
        
        with gr.Column():
            transcription_output = gr.Textbox(label="Transcription")
            score_output = gr.Number(label="Pronunciation Score")
            feedback_output = gr.Textbox(label="Feedback")

    submit_button.click(mispronunciation_detection, 
                        inputs=[audio_input, standard_text_input], 
                        outputs=[transcription_output, score_output, feedback_output])

demo.launch()