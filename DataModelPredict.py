from pydantic import BaseModel
from typing import Optional

class DataModelPredict(BaseModel):

# Estas varibles permiten que la librería pydantic haga el parseo entre el Json recibido y el modelo declarado.
    Textos_espanol: str
    sdg: Optional[int] = None


#Esta función retorna los nombres de las columnas correspondientes con el modelo exportado en joblib.
    def columns(self):
        return [
            "Textos_espanol",
            "sdg"
        ]
