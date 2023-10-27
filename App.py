import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import torch
from pydub import AudioSegment
from pydub.playback import play
import tempfile
import numpy as np
import os

from ImportScript import load_models

@st.cache_resource
def load_and_create_models():
    classification_model, classification_tokenizer, music_processor, music_model, loaded_poem_generator = load_models()
    return classification_model, classification_tokenizer, music_processor, music_model, loaded_poem_generator

classification_model, classification_tokenizer, music_processor, music_model, loaded_poem_generator = load_and_create_models()

def predict_sentiment(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    sentiment = outputs.logits.argmax().item()
    
    if sentiment == 0:
        return "Negative"
    elif sentiment == 1:
        return "Neutral"
    elif sentiment == 2:
        return "Positive"
    else:
        return "Unknown"


def generate_music(text, processor, model, max_new_tokens=256):

    inputs = processor(
        text=[text],
        padding=True,
        return_tensors="pt",
    )
    audio_values = model.generate(**inputs, max_new_tokens=max_new_tokens)
    sampling_rate = model.config.audio_encoder.sampling_rate

    return audio_values, sampling_rate


def generate_poem(input_text, model=loaded_poem_generator):
    mapping = {
               "Positive":"<happy>","neutral":"<neutral>","Negative":"<sad>","Unknown":"<surprise>"
              }
    
    replaced_text = mapping.get(input_text, input_text)
    input_prompt = "<BOS>" + " " + replaced_text
    

    poem = model(input_prompt, max_length=200, do_sample=True,
               repetition_penalty=1.1, temperature=1.2,
               top_p=0.95, top_k=50)
    
    return poem[0]["generated_text"]

st.title("Sentiment Analyzer")

st.sidebar.header('Enter the text you want to analyze:')

user_input = st.sidebar.text_area("")
if st.button("Analyze Sentiment"):
  if user_input:
      sentiment = predict_sentiment(user_input, classification_tokenizer, classification_model)
      if sentiment == "Negative":
        st.write("You look sad 😔, every thing will be ok ❤️")
      elif sentiment == "Positive":
        st.write("You look happy 😃, Your positivity and smile are contagious, spreading happiness wherever you go. ❤️")
      else:
        st.write("You looks neither happy nor sad  🙂, I hope today brings you reasons to smile and moments of joy. Embrace the positivity around you and let it brighten your day. ❤️")

      audio_values, sampling_rate = generate_music(sentiment, music_processor, music_model, max_new_tokens=256)
      
      audio_data = audio_values[0].cpu().numpy()

      with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
          temp_wav_name = temp_wav.name
          audio_data = audio_data * (2**15 - 1) 
          audio_data = audio_data.astype(np.int16)
          temp_wav.write(audio_data.tobytes())

      st.audio(audio_values[0].numpy(), format="audio/wav", sample_rate=sampling_rate)

      poem = generate_poem(sentiment)
      st.markdown(
          f"<div style='border: 2px solid #102C57; padding: 10px; text-align: center; font-size: 18px;'>{poem}</div>",
          unsafe_allow_html=True
      )
