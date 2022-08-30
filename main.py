from typing import Union

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

import torch
from transformers import BertForSequenceClassification, AutoTokenizer

LABELS = ['neutral', 'happiness', 'sadness', 'enthusiasm', 'fear', 'anger', 'disgust']
tokenizer = AutoTokenizer.from_pretrained('Aniemore/rubert-tiny2-russian-emotion-detection')
model = BertForSequenceClassification.from_pretrained('Aniemore/rubert-tiny2-russian-emotion-detection')


app = FastAPI()


class Item(BaseModel):
    name: str
    price: float
    is_offer: Union[bool, None] = None


@torch.no_grad()
def predict_emotions(text: str) -> list:
    """
        It takes a string of text, tokenizes it, feeds it to the model, and returns a dictionary of emotions and their
        probabilities
        :param text: The text you want to classify
        :type text: str
        :return: A dictionary of emotions and their probabilities.
    """
    inputs = tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**inputs)
    predicted = torch.nn.functional.softmax(outputs.logits, dim=1)
    emotions_list = {}
    for i in range(len(predicted.numpy()[0].tolist())):
        emotions_list[LABELS[i]] = predicted.numpy()[0].tolist()[i]
    return emotions_list


def format(result: dict):
    results = []
    for i in result.items():
        results.append({'label': i[0], 'score': i[1]})
    return results

# @app.get("/")
# def read_root():
#     return {"Hello": "World"}
#
#
# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Union[str, None] = None):
#     return {"item_id": item_id, "q": q}


@app.post("/string/{string}")
def update_item(string: str):
    return {'results': format(predict_emotions(string))}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
