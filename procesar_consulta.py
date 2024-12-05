import json
import os
import re
import numpy as np
from pymongo import MongoClient
from transformers import BertTokenizer, BertModel
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

# RUTA_EMBEDDINGS debería ser cargado una vez al inicio
RUTA_EMBEDDINGS = r'C:\Users\56974\Desktop\seminario 2024\codigos python avanzados\embeddings_avanzado_test.npy'
EMBEDDINGS_DICT = np.load(RUTA_EMBEDDINGS, allow_pickle=True).item()

# Configuración de la base de datos MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['indice_invertido_decretos_munvalp_test']
collection = db['indice_invertido_test']

stop_words = set(stopwords.words('spanish'))

def preprocesar_consulta(query):
    query = query.lower()
    query = re.sub(r'[^\w\s]', '', query)
    terms = query.split()
    terms = [word for word in terms if word not in stop_words]
    return terms

def buscar_en_indice_invertido_multiple(terms):
    doc_lists = []
    for term in terms:
        cursor = collection.find({"word": term})
        files = set()
        for resultado in cursor:
            if 'documents' in resultado:
                for doc in resultado['documents']:
                    if isinstance(doc, dict):
                        files.add(doc.get('documento', 'Documento desconocido'))
                    elif isinstance(doc, str):
                        files.add(doc)
        if files:
            doc_lists.append(files)

    if not doc_lists:
        return []

    doc_intersection = set.intersection(*doc_lists) if len(doc_lists) > 1 else doc_lists[0]
    return list(doc_intersection) if doc_intersection else list(set.union(*doc_lists))

def obtener_embeddings(consulta, modelo, tokenizador):
    inputs = tokenizador(consulta, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = modelo(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

def calcular_tfidf(corpus, query):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    return vectorizer, X, vectorizer.transform([query])

# Función para calcular la magnitud de un vector
def calcular_magnitud(vector):
    return np.sqrt(np.sum(vector ** 2))

# Función para calcular la similitud del coseno entre dos vectores
def similitud_coseno(vector1, vector2):
    producto_punto = np.dot(vector1, vector2)
    magnitud_vector1 = calcular_magnitud(vector1)
    magnitud_vector2 = calcular_magnitud(vector2)
    if magnitud_vector1 == 0 or magnitud_vector2 == 0:
        return 0  # Evitar división por cero
    return producto_punto / (magnitud_vector1 * magnitud_vector2)

# Cargar el modelo BERT y el tokenizador una sola vez
tokenizador = BertTokenizer.from_pretrained('bert-base-uncased')
modelo = BertModel.from_pretrained('bert-base-uncased')

def run(query: str):
    """Función principal que procesa la consulta y devuelve los documentos relevantes."""
    try:
        query = query.lower()
        terms = preprocesar_consulta(query)
        doc_ids = buscar_en_indice_invertido_multiple(terms)

        if not doc_ids:
            return []

        # Obtener embedding de la consulta
        try:
            embedding_consulta = obtener_embeddings(query, modelo, tokenizador)
        except Exception as e:
            raise RuntimeError(f"Error al obtener embeddings de la consulta: {str(e)}")

        # Verificación del corpus
        try:
            corpus = [os.path.splitext(doc_id)[0] for doc_id in doc_ids if EMBEDDINGS_DICT.get(os.path.splitext(doc_id)[0]) is not None and EMBEDDINGS_DICT.get(os.path.splitext(doc_id)[0]).size > 0]
            if not corpus:
                return []
        except Exception as e:
            raise RuntimeError(f"Error al procesar el corpus: {str(e)}")

        # Calcular TF-IDF
        try:
            vectorizer, X, _ = calcular_tfidf(corpus, query)
        except Exception as e:
            raise RuntimeError(f"Error al calcular TF-IDF: {str(e)}")

        resultados = []
        for i, doc_id in enumerate(doc_ids):
            base_doc_id = os.path.splitext(doc_id)[0]
            embedding_doc = EMBEDDINGS_DICT.get(base_doc_id)

            if embedding_doc is not None and embedding_doc.size > 0:
                try:
                    if np.any(embedding_consulta) and np.any(embedding_doc):
                        # Usar la similitud del coseno
                        similitud = similitud_coseno(embedding_consulta, embedding_doc)
                        tfidf_score = X[i].todense().tolist()
                        resultados.append({
                            '_id': doc_id,
                            'similitud': similitud,
                            'palabras_utilizadas': query,
                            'tfidf_documento': tfidf_score
                        })
                except Exception as e:
                    raise RuntimeError(f"Error al calcular similitud para {doc_id}: {str(e)}")

        resultados_ordenados = sorted(resultados, key=lambda x: x['similitud'], reverse=True)

        # Imprimir los resultados para verificar antes de retornar
        print(f"Resultados ordenados: {resultados_ordenados}")

        return resultados_ordenados

    except Exception as e:
        print(f"Error: {str(e)}")
        return {"error": str(e)}
