import os
import json
import re
import pytesseract
import time
from pdf2image import convert_from_path
from PIL import Image
from nltk.corpus import stopwords
from pymongo import MongoClient
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import log
from pymongo.errors import ConnectionFailure

# Cargar stopwords en español y agregar letras individuales a eliminar
stop_words = set(stopwords.words('spanish'))
stop_words.update(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'])

def pdf_to_text(pdf_path):
    """Convertir PDF a texto usando OCR y contar las páginas."""
    images = convert_from_path(pdf_path)
    text = ''
    for image in images:
        text += pytesseract.image_to_string(image)
    return text, len(images)  # Retorna el texto y el número de páginas

def extract_norma_number_and_year(filename):
    """Extraer el número de norma y el año del nombre del archivo."""
    match = re.search(r'Decreto_Nº_(\d+)_del_(\d{4})', filename)
    if match:
        norma_number = match.group(1)
        year = match.group(2)
        return norma_number, year
    return None, None

def process_pdf_file(pdf_path):
    """Procesar un archivo PDF y devolver las palabras utilizadas, eliminadas y el número de páginas."""
    text, num_pages = pdf_to_text(pdf_path)
    text = text.lower()
    words = re.findall(r'\b[a-záéíóúñü]+\b', text)
    used_words = [word for word in words if word not in stop_words]
    removed_words = [word for word in words if word in stop_words]
    return used_words, removed_words, num_pages

def build_inverted_index_parallel(folder_path, used_words_path, removed_words_path, stats_path, processing_times_path):
    """Construir el índice invertido, calcular TF-IDF, contar palabras utilizadas y eliminadas, y calcular tiempos de procesamiento."""
    inverted_index = {}
    doc_count = 0
    word_doc_count = {}
    total_used_words = 0
    total_removed_words = 0
    total_processing_time = 0
    total_pages = 0
    total_words_per_page = 0  # Para calcular la cantidad promedio de palabras por hoja

    pdf_files = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith('.pdf')]
    total_files = len(pdf_files)

    # Almacenar palabras utilizadas, eliminadas, y tiempos de procesamiento
    all_used_words = []
    all_removed_words = []
    processing_times = []  # Lista para almacenar tiempos de procesamiento de cada archivo

    start_time = time.time()

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_pdf_file, pdf_path): pdf_path for pdf_path in pdf_files}
        for i, future in enumerate(as_completed(futures)):
            pdf_path = futures[future]
            filename = os.path.basename(pdf_path)
            doc_id = os.path.splitext(filename)[0]
            norma_number, year = extract_norma_number_and_year(filename)
            try:
                # Medir tiempo de procesamiento por archivo
                file_start_time = time.time()
                used_words, removed_words, num_pages = future.result()
                file_end_time = time.time()
                processing_time = file_end_time - file_start_time

                # Evitar dividir por cero en el cálculo de tiempo por hoja
                avg_time_per_page = processing_time / num_pages if num_pages > 0 else 0

                # Registrar información del procesamiento de este archivo
                processing_times.append({
                    "file": filename,
                    "total_time_seconds": round(processing_time, 6),
                    "pages": num_pages,
                    "avg_time_per_page_seconds": round(avg_time_per_page, 6)
                })

                # Acumular tiempos y páginas totales
                total_processing_time += processing_time
                total_pages += num_pages

                # Acumular palabras utilizadas y eliminadas
                all_used_words.extend(used_words)
                all_removed_words.extend(removed_words)
                total_used_words += len(used_words)
                total_removed_words += len(removed_words)
                total_words_per_page += (len(used_words) + len(removed_words)) / num_pages if num_pages > 0 else 0
                doc_count += 1

                word_counts = {word: used_words.count(word) for word in set(used_words)}
                total_words = len(used_words)

                for word, count in word_counts.items():
                    if word not in inverted_index:
                        inverted_index[word] = []
                    tf = count / total_words

                    inverted_index[word].append({
                        "documento": filename,
                        "numero_norma": norma_number,
                        "tf": tf,
                        "fecha": year,
                        "estado": "activo"
                    })

                    # Actualizar el contador de documentos para IDF
                    word_doc_count[word] = word_doc_count.get(word, 0) + 1

                print(f"Archivo '{filename}' procesado en {processing_time:.10f} segundos, {num_pages} páginas, promedio por hoja: {avg_time_per_page:.10f} segundos")

            except Exception as e:
                print(f"Error procesando {filename}: {e}")

    # Calcular el IDF de cada palabra y actualizar el índice con TF-IDF
    for word, docs in inverted_index.items():
        idf = log(doc_count / word_doc_count[word])
        for entry in docs:
            entry["tf_idf"] = entry["tf"] * idf

    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Tiempo total de ejecución: {int(hours):02}:{int(minutes):02}:{int(seconds):02} (hh:mm:ss)")
    print(f"Tiempo total de ejecución: {elapsed_time:.2f} segundos")
    print(f"Cantidad de archivos procesados: {total_files}")
    print(f"Total de palabras utilizadas: {total_used_words}")
    print(f"Total de palabras eliminadas: {total_removed_words}")

    # Calcular promedios
    overall_avg_time_per_page = round(total_processing_time / total_pages, 6) if total_pages > 0 else 0
    avg_words_per_page = round(total_words_per_page / total_files, 6) if total_files > 0 else 0
    avg_words_per_document = round((total_used_words + total_removed_words) / total_files, 6) if total_files > 0 else 0
    avg_pages_per_document = round(total_pages / total_files, 6) if total_files > 0 else 0

    # Guardar estadísticas en JSON
    stats = {
        "total_used_words": total_used_words,
        "total_removed_words": total_removed_words,
        "overall_avg_time_per_page": overall_avg_time_per_page,
        "avg_words_per_page": avg_words_per_page,
        "avg_words_per_document": avg_words_per_document,
        "avg_pages_per_document": avg_pages_per_document
    }
    with open(stats_path, 'w', encoding='utf-8') as stats_file:
        json.dump(stats, stats_file, ensure_ascii=False, indent=4)
    print(f"Estadísticas guardadas en {stats_path}")

    # Guardar palabras utilizadas y eliminadas en JSON
    with open(used_words_path, 'w', encoding='utf-8') as used_file:
        json.dump(all_used_words, used_file, ensure_ascii=False, indent=4)
    with open(removed_words_path, 'w', encoding='utf-8') as removed_file:
        json.dump(all_removed_words, removed_file, ensure_ascii=False, indent=4)
    print(f"Palabras utilizadas guardadas en {used_words_path}")
    print(f"Palabras eliminadas guardadas en {removed_words_path}")

    # Guardar tiempos de procesamiento de cada archivo en JSON
    with open(processing_times_path, 'w', encoding='utf-8') as times_file:
        json.dump(processing_times, times_file, ensure_ascii=False, indent=4)
    print(f"Tiempos de procesamiento guardados en {processing_times_path}")

    return inverted_index

def save_inverted_index_to_json(inverted_index, output_path):
    """Guardar el índice invertido en un archivo JSON."""
    sorted_index = dict(sorted(inverted_index.items()))
    with open(output_path, 'w', encoding='utf-8') as json_file:
        json.dump(sorted_index, json_file, ensure_ascii=False, indent=4)

def save_inverted_index_to_mongodb(inverted_index, db_name, collection_name):
    """Guardar el índice invertido en MongoDB."""
    try:
        client = MongoClient('mongodb://localhost:27017/')
        client.admin.command('ping')
        
        db = client[db_name]
        collection = db[collection_name]
        
        documents = [{'word': word, 'documents': docs} for word, docs in inverted_index.items()]
        if documents:
            collection.insert_many(documents)
            print(f"{len(documents)} documentos insertados en MongoDB en la colección '{collection_name}'")
        else:
            print("No hay documentos para insertar en MongoDB.")
            
    except ConnectionFailure:
        print("Error: No se pudo conectar a MongoDB. Verifique la conexión.")
    except Exception as e:
        print(f"Error al insertar documentos en MongoDB: {e}")

def load_json_to_mongodb(json_path, db_name, collection_name):
    """Cargar un archivo JSON en MongoDB."""
    try:
        client = MongoClient('mongodb://localhost:27017/')
        db = client[db_name]
        collection = db[collection_name]
        
        with open(json_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
        
        documents = [{'word': word, 'documents': docs} for word, docs in data.items()]
        
        if documents:
            collection.insert_many(documents)
            print(f"{len(documents)} documentos insertados desde JSON a MongoDB en la colección '{collection_name}'")
        else:
            print("No hay documentos para insertar en MongoDB.")
            
    except Exception as e:
        print(f"Error al cargar JSON a MongoDB: {e}")

# Configurar paths y conexión a MongoDB
folder_path = r"C:\Users\56974\Desktop\seminario 2024\codigos python avanzados\decretos_2023_test"
output_path = r"C:\Users\56974\Desktop\seminario 2024\codigos python avanzados\indice_invertido_con_stopwords_normalizados_avanzado_test.json"
used_words_path = r"C:\Users\56974\Desktop\seminario 2024\codigos python avanzados\used_words_test.json"
removed_words_path = r"C:\Users\56974\Desktop\seminario 2024\codigos python avanzados\removed_words_test.json"
stats_path = r"C:\Users\56974\Desktop\seminario 2024\codigos python avanzados\stats_test.json"
processing_times_path = r"C:\Users\56974\Desktop\seminario 2024\codigos python avanzados\processing_times_test.json"
db_name = 'indice_invertido_decretos_munvalp_test'
collection_name = 'indice_invertido_test'

# Verificar si el archivo JSON ya existe y está completo
if os.path.exists(output_path):
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            json.load(f)
        load_json_to_mongodb(output_path, db_name, collection_name)
    except (json.JSONDecodeError, FileNotFoundError):
        inverted_index = build_inverted_index_parallel(folder_path, used_words_path, removed_words_path, stats_path, processing_times_path)
        save_inverted_index_to_json(inverted_index, output_path)
        save_inverted_index_to_mongodb(inverted_index, db_name, collection_name)
else:
    inverted_index = build_inverted_index_parallel(folder_path, used_words_path, removed_words_path, stats_path, processing_times_path)
    save_inverted_index_to_json(inverted_index, output_path)
    save_inverted_index_to_mongodb(inverted_index, db_name, collection_name)
