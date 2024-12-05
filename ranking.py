import sys
import json
import numpy as np
import math
from collections import Counter
import os
from transformers import BertTokenizer, BertModel

# Ruta de los archivos de documentos
RUTA_DOCUMENTOS = r'C:\Users\56974\Desktop\seminario 2024\codigos python avanzados\decretos_2023_test'

# Ruta del archivo de embeddings precalculados
RUTA_EMBEDDINGS = r'C:\Users\56974\Desktop\seminario 2024\codigos python avanzados\embeddings_avanzado_test.npy'

if len(sys.argv) < 2:
    print("Error: No se proporcionaron la consulta o la ruta de documentos.")
    sys.exit(1)

query = sys.argv[1]

# Función para obtener embeddings de BERT
def obtener_embeddings(texto, modelo, tokenizador):
    inputs = tokenizador(texto, return_tensors='pt', truncation=True, padding=True)
    outputs = modelo(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# Inicializar BERT
tokenizador = BertTokenizer.from_pretrained('bert-base-uncased')
modelo = BertModel.from_pretrained('bert-base-uncased')

try:
    # Cargar los embeddings precalculados
    embeddings_dict = np.load(RUTA_EMBEDDINGS, allow_pickle=True).item()

    # Obtener embedding de la consulta
    embedding_consulta = obtener_embeddings(query, modelo, tokenizador)

    # Calcular similitudes
    resultados = []
    for doc_id, embedding_doc in embeddings_dict.items():
        similitud = (embedding_consulta @ embedding_doc.T).flatten()[0]
        resultados.append({'_id': doc_id, 'similitud': float(similitud)})

    # Ordenar por similitud
    resultados_ordenados = sorted(resultados, key=lambda x: x['similitud'], reverse=True)
    print(json.dumps(resultados_ordenados))

except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)