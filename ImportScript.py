import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import torch
from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    GPT2LMHeadModel,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    LineByLineTextDataset,
    PreTrainedTokenizer,
    TextDataset,
    Trainer,
    TrainingArguments,
    set_seed,
    pipeline,
    TextGenerationPipeline,
    GPT2LMHeadModel,
    AutoTokenizer
)

def load_models():
    classification_model_name = "finiteautomata/bertweet-base-sentiment-analysis"
    classification_tokenizer = AutoTokenizer.from_pretrained(classification_model_name)
    classification_model = AutoModelForSequenceClassification.from_pretrained(classification_model_name)

    music_model_name = "facebook/musicgen-small"
    music_processor = AutoProcessor.from_pretrained(music_model_name)
    music_model = MusicgenForConditionalGeneration.from_pretrained(music_model_name)

    poem_generation_model_path = "/content/drive/MyDrive/expression_poem_generator/Poem_Generation/Tuned_model"
    poem_generation_model = GPT2LMHeadModel.from_pretrained(poem_generation_model_path)
    poem_generation_tokenizer = AutoTokenizer.from_pretrained(poem_generation_model_path)
    loaded_poem_generator = TextGenerationPipeline(model=poem_generation_model, tokenizer=poem_generation_tokenizer)

    return classification_model, classification_tokenizer, music_processor, music_model, loaded_poem_generator