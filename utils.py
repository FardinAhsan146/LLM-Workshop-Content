import json
import os 
import pandas as pd 
import numpy as np 
import requests

from typing import List
from dotenv import load_dotenv
load_dotenv() 

api_key = os.environ['OPENAI_API_KEY']

def read_file(file_path: str) -> str:
    """ Kinda obvious isn't it? """
    with open(file_path, 'r', encoding='utf-8') as file:
        contents = file.read()
    return contents

def cosine_similarity(a:list, b:list) -> float:
    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = sum(x * x for x in a) ** 0.5
    magnitude_b = sum(x * x for x in b) ** 0.5
    return dot_product / (magnitude_a * magnitude_b)

def get_embedding(text_input:str) -> List[float]:
    """
    https://platform.openai.com/docs/guides/embeddings/what-are-embeddings
    """
    # Request headers, body, and URL
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    data = {
        "input": text_input,
        "model": "text-embedding-ada-002"
    }
    url = 'https://api.openai.com/v1/embeddings'
    
    # Get response and return the relevant parts only
    response = requests.post(url, headers=headers, json=data)
    return response.json()['data'][0]['embedding']

def get_openai_chat(message_list:list) -> str:
    """
    https://platform.openai.com/docs/api-reference/chat/create
    """
    api_url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]
    }
    data = {
        "model": "gpt-3.5-turbo",
        "messages": message_list
    }

    response = requests.post(api_url, headers=headers, data=json.dumps(data))
    return response.json()['choices'][0]['message']['content']


def chat_with_gpt() -> None:
    """lol"""
    messages = [{"role": "system","content": "You are a helpful assistant."},{"role": "assistant","content": "Hello!"}]

    while True: 
        user_input = input("User: ")
        if user_input == "STOP":
            break

        messages.append({'role':'user','content':user_input})
        result = get_openai_chat(messages)
        print(f"Bot: {result}")
        messages.append({'role':'assistant','content':result})