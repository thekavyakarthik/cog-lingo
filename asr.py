from transformers import AutoModelForCTC, Wav2Vec2Processor

model = AutoModelForCTC.from_pretrained("kavsss/asr-for-phoneme-detection-1")
processor = Wav2Vec2Processor.from_pretrained("kavsss/asr-for-phoneme-detection-1")

import torch
from phoneme import extract_needed_data, speech_file_to_array
from analysis import calculate_confidence

from gtts import gTTS
import os
import numpy as np

def prepare_dataset(audio_arr):
    if isinstance(audio_arr, list):
        audio_arr = np.concatenate(audio_arr) 
    batch = processor(audio_arr, sampling_rate=16000).input_values[0]
    return batch

def predict_phoneme(batch):
    with torch.no_grad():
       input_values = torch.tensor(batch, device="cpu").unsqueeze(0)
       logits = model(input_values).logits
    
    pred_ids = torch.argmax(logits, dim=-1)
    pred_phoneme = processor.batch_decode(pred_ids)[0]
    
    return pred_phoneme

def text_to_speech(text):
    # Initialize gTTS with the text to convert
    speech = gTTS(text)

    # Save the audio file to a temporary file
    speech_file = 'speech.wav'
    speech.save(speech_file)

    return speech_file


def score(audio_arr, text):
    batch = prepare_dataset(audio_arr)
    pred = predict_phoneme(batch)

    real_file_path = text_to_speech(text)
    real_audio_arr = speech_file_to_array(real_file_path)
    batch = prepare_dataset(real_audio_arr)
    real = predict_phoneme(batch)

    confidence = calculate_confidence(real, pred)
    return pred, round(confidence*100, 2), "feedback ai to be implemented"
    

#with processor.as_target_processor():
#  real = processor(batch["phoneme"]).input_ids