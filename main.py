from typing import List, Optional

from fastapi import FastAPI

from DataModelRetrain import DataModelRetrain
from DataModelPredict import DataModelPredict
from fastapi.middleware.cors import CORSMiddleware

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import pandas as pd

import joblib

app = FastAPI()

# Configurar CORS
origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import sys
from transformer import TextNormalizer

sys.modules['__main__'] = sys.modules['transformer']

df_old = pd.read_excel('assets/ODScat_345.xlsx')

# Variable para almacenar los nuevos datos en la sesión
df_new_session = pd.DataFrame()


@app.get("/")
def read_root():
   return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
   return {"item_id": item_id, "q": q}

@app.post("/predict")
def make_predictions(dataModels: List[DataModelPredict]):
    data_dicts = [dataModel.dict() for dataModel in dataModels]
    df = pd.DataFrame(data_dicts)
    model = joblib.load("assets/modelo.joblib")
    predictions = model.predict(df["Textos_espanol"])
    probabilities = model.predict_proba(df["Textos_espanol"])
    response = []
    for pred, prob in zip(predictions, probabilities):
        response.append({
            "prediction": int(pred),
            "probability": float(max(prob))
        })
    return response

@app.post("/re-train")
def retrain_model(dataModel: List[DataModelRetrain]):
    global df_new_session
    # Convertir la lista de DataModel a DataFrame
    df = pd.DataFrame([data.dict() for data in dataModel])
    # Almacenar los nuevos datos en la sesión
    df_new_session = pd.concat([df_new_session, df], ignore_index=True)
    # Combinar los datos antiguos con los nuevos de la sesión
    df_combined = pd.concat([df_old, df_new_session], ignore_index=True)
    # Separar las características y la variable objetivo
    X = df_combined['Textos_espanol']
    y = df_combined['sdg']

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    # Cargar el modelo entrenado
    model = joblib.load("assets/modelo.joblib")
    # Re-entrenar el modelo con los nuevos datos
    model.fit(X_train, y_train)
    # Guardar el modelo reentrenado
    joblib.dump(model, 'assets/modelo.joblib')

    # Hacer predicciones en el conjunto de prueba
    y_pred = model.predict(X_test)

    # Calcular las métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    }