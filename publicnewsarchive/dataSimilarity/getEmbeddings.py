import hashlib
from typing import List
from transformers import AutoTokenizer, AutoModel
import torch
import json

def load_model(model_name: str):
    '''
    Function to load a pre-trained model and tokenizer
    
    Args:
    model_name: str: name of the pre-trained model
    '''
    
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model.eval()
    return model, tokenizer

def create_embedding(text: str, model, tokenizer) -> List[float]:
    '''
    Function to create the embedding for a given text using a pre-trained model
    
    Args:
    text: str: text for which the embedding is to be created
    model: model object: pre-trained model object
    tokenizer: tokenizer object: pre-trained tokenizer object
    '''
    
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # using the mean of the last hidden states as the embedding
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

    return embedding

def save_embedding(embedding: List[float], text: str, file_path: str):
    '''
    Function to save the embedding for a given text
    
    Args:
    embedding: List[float]: embedding of the text
    text: str: text for which the embedding is created
    file_path: str: path to the file where the embeddings are to be saved
    '''
    
    try:
        with open(file_path, 'r') as f:
            embeddings_dict = json.load(f)
    except FileNotFoundError:
        embeddings_dict = {}
    
    text_hash = hashlib.md5(text.encode()).hexdigest()
    embeddings_dict[text_hash] = embedding
    
    with open(file_path, 'w') as f:
        json.dump(embeddings_dict, f, indent=4)
