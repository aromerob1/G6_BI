from sklearn.base import BaseEstimator, TransformerMixin


from sklearn.base import BaseEstimator, TransformerMixin
import spacy

class TextNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self, stop_words):
        self.stop_words = stop_words
        self.nlp = None 

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.nlp = None

    def normalize_document(self, doc):
        if self.nlp is None:
            self.nlp = spacy.load('es_core_news_sm')
        # Convertir a minúsculas y eliminar espacios
        doc = doc.lower().strip()
        # Procesar el documento con spaCy
        spacy_doc = self.nlp(doc)
        # Lematización y eliminación de stopwords
        lemmatized_tokens = [token.lemma_ for token in spacy_doc if token.text not in self.stop_words]
        # Reconstruir el texto lematizado
        return ' '.join(lemmatized_tokens)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return [self.normalize_document(doc) for doc in X]
