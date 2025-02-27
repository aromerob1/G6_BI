from joblib import load

from transformer import TextNormalizer


class Model:

    def __init__(self,columns):
        self.model = load("assets/modelo.joblib")

    def make_predictions(self, data):
        result = self.model.predict(data)
        return result
