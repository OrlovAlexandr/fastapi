from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel				# pydantic занимается автоматической проверкой формата и типа данных
											# Импортировали вариант класса для моделей BaseModel 
class Item(BaseModel):						# создаем на его основе свою модель
    text: str 								# В параметре text будем передавать текст для определения тональности

app = FastAPI()
classifier = pipeline("sentiment-analysis")

@app.get("/")
def root():
    return {"message": "Hello World"}

@app.post("/predict/")						# метод POST для больших текстов в теле запроса HTTP
def predict(item: Item):					# в predict передается объект item типа Item
    """Sentiment analysis for a text"""
    return classifier(item.text )[0]		# классификатор вызывается для поля text объекта item
