from pydantic import BaseModel

class DataModelRetrain(BaseModel):

# Estas varibles permiten que la librería pydantic haga el parseo entre el Json recibido y el modelo declarado.
    Textos_espanol: str
    sdg: int


#Esta función retorna los nombres de las columnas correspondientes con el modelo exportado en joblib.
    def columns(self):
        return [
            "Textos_espanol",
            "sdg"
        ]
