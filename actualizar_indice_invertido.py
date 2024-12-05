import json
import os
from nltk.corpus import stopwords
from config_db import collection  # Importar colección desde config_db

# Ruta del archivo JSON del índice invertido
JSON_PATH = r'C:\Users\56974\Desktop\seminario 2024\codigos python avanzados\indice_invertido_con_stopwords_normalizados_avanzado_test.json'
stop_words = set(stopwords.words('spanish'))

# Función para cargar el índice invertido desde un archivo JSON
def cargar_indice(json_path):
    #print(f"[DEBUG] Cargando índice desde {json_path}")
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    #print("[DEBUG] No se encontró un índice existente. Creando uno nuevo.")
    return {}

# Función para guardar el índice invertido en un archivo JSON
def guardar_indice(json_path, indice_invertido):
    #print(f"[DEBUG] Guardando índice en {json_path}")
    with open(json_path, 'w', encoding='utf-8') as file:
        json.dump(indice_invertido, file, ensure_ascii=False, indent=4)

# Función para extraer información de norma y año
def extract_norma_number_and_year(filename):
    #print(f"[DEBUG] Extrayendo norma y año de {filename}")
    parts = filename.split('_')
    norma_number = parts[2]
    year = parts[-1].split('.')[0]
    #print(f"[DEBUG] Norma: {norma_number}, Año: {year}")
    return norma_number, year

# Función para actualizar el índice invertido
def actualizar_indice_invertido(filename, words, json_path):
    #print(f"[DEBUG] Entrando en actualizar_indice_invertido para {filename}")
    #print(f"[DEBUG] Tipo de 'collection' al entrar: {type(collection)}")
    indice_invertido = cargar_indice(json_path)
    #print(f"[DEBUG] Índice cargado. Total de términos: {len(indice_invertido)}")

    for item in words:
        if item not in stop_words and item.strip():
            #print(f"[DEBUG] Procesando término: {item}")
            if item not in indice_invertido:
                #print(f"[DEBUG] Añadiendo nuevo término al índice: {item}")
                indice_invertido[item] = []
            
            if not any(doc.get('documento') == filename for doc in indice_invertido[item]):
                norma_number, year = extract_norma_number_and_year(filename)
                doc_entry = {
                    "documento": filename,
                    "numero_norma": norma_number,
                    "tf": words.count(item) / len(words),
                    "fecha": year,
                    "estado": "activo"
                }
                indice_invertido[item].append(doc_entry)
                #print(f"[DEBUG] Documento añadido al índice para término: {item}")

                # Actualizar MongoDB
                collection.update_one(
                    {"word": item},
                    {"$addToSet": {"documents": doc_entry}},
                    upsert=True
                )
                #print(f"[DEBUG] MongoDB actualizado para término: {item}")
    
    guardar_indice(json_path, indice_invertido)
   # print(f"[DEBUG] Índice invertido actualizado para {filename}")
