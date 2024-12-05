import os
import time
from threading import Thread, Lock
from concurrent.futures import ThreadPoolExecutor
from actualizar_indice_invertido import actualizar_indice_invertido
from actualizar_embeddings import actualizar_embeddings

# Configuraciones y rutas
RUTA_DOCUMENTOS = r'C:\Users\56974\Desktop\seminario 2024\codigos python avanzados\decretos_2023_test'
OUTPUT_PATH = r'C:\Users\56974\Desktop\seminario 2024\codigos python avanzados\indice_invertido_con_stopwords_normalizados_avanzado_test.json'
EMBEDDINGS_PATH = r'C:\Users\56974\Desktop\seminario 2024\codigos python avanzados\embeddings_avanzado.npy'
CHECK_INTERVAL = 120  # Intervalo de verificación en segundos
ARCHIVO_PROCESADOS = 'archivos_procesados.txt'
lock = Lock()

# Cargar lista de archivos procesados
def cargar_archivos_procesados():
    if os.path.exists(ARCHIVO_PROCESADOS):
        with open(ARCHIVO_PROCESADOS, 'r', encoding='utf-8') as f:
            return set(line.strip() for line in f.readlines())
    return set()

# Registrar un archivo como procesado
def guardar_archivo_procesado(filename):
    with open(ARCHIVO_PROCESADOS, 'a', encoding='utf-8') as f:
        f.write(f"{filename}\n")

# Procesar un documento PDF
def procesar_documento(pdf_path, filename, output_path, embeddings_path):
    try:
        print(f"[INFO] Procesando nuevo archivo: {filename}")
        # Actualizar índice invertido
        actualizar_indice_invertido(pdf_path, filename, output_path)

        # Actualizar embeddings
        actualizar_embeddings(pdf_path, filename, embeddings_path)

        # Registrar archivo como procesado
        guardar_archivo_procesado(filename)
        print(f"[INFO] Archivo procesado correctamente: {filename}")
    except Exception as e:
        print(f"[ERROR] Error procesando {filename}: {e}")

# Verificar y procesar nuevos archivos
def check_for_new_files(folder_path, output_path, embeddings_path):
    archivos_procesados = cargar_archivos_procesados()
    current_files = set(f for f in os.listdir(folder_path) if f.endswith('.pdf'))
    nuevos_archivos = current_files - archivos_procesados

    if nuevos_archivos:
        print(f"[INFO] Detectados {len(nuevos_archivos)} nuevos archivos.")
        with ThreadPoolExecutor(max_workers=4) as executor:
            for filename in nuevos_archivos:
                pdf_path = os.path.join(folder_path, filename)
                executor.submit(procesar_documento, pdf_path, filename, output_path, embeddings_path)
    else:
        print("[INFO] No se detectaron nuevos archivos.")

# Iniciar el crawler en un hilo continuo
def start_crawler(folder_path, output_path, embeddings_path):
    while True:
        check_for_new_files(folder_path, output_path, embeddings_path)
        time.sleep(CHECK_INTERVAL)

if __name__ == '__main__':
    print("[INFO] Iniciando Crawler...")

    # El programa solo escucha archivos nuevos; no procesa los existentes.
    thread = Thread(target=start_crawler, args=(RUTA_DOCUMENTOS, OUTPUT_PATH, EMBEDDINGS_PATH))
    thread.daemon = True
    thread.start()

    # Mantener el programa en ejecución
    while True:
        time.sleep(1)
