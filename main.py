from fastapi import FastAPI
from pydantic import BaseModel  # pydantic занимается автоматической проверкой
# формата и типа данных
from transformers import pipeline


# Импортировали вариант класса для моделей BaseModel.
# Не-а херь. Этот класс наследует pydantic, который занимается валидацией того,
# что то, что подается на веб-серввер точно является строкой, или даже
# преобразует в строку
class Item(BaseModel):
    text: str  # В параметре text будем передавать текст для определения тональности


app = FastAPI()  # создаем на его основе свою модель
classifier = pipeline("sentiment-analysis")


@app.get("/")
def root():
    return {"message": "Hello World"}


@app.post("/predict/")  # метод POST для больших текстов в теле запроса HTTP
def predict(item: Item):  # в predict передается объект item типа Item
    return classifier(item.text)[0]  # классификатор вызывается для поля text
                                     # объекта item
