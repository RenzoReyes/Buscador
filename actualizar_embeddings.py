import os
import numpy as np
import pytesseract
from pdf2image import convert_from_path
from transformers import BertTokenizer, BertModel
import torch

# Ruta del archivo binario con los embeddings
OUTPUT_PATH = r'C:\Users\56974\Desktop\seminario 2024\codigos python avanzados\embeddings_avanzado_test.npy'

# Inicializar el modelo y tokenizador BETO
tokenizador = BertTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased')
modelo = BertModel.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased')
modelo.eval()

# Función para cargar embeddings existentes
def cargar_embeddings(output_path):
    if os.path.exists(output_path):
        return np.load(output_path, allow_pickle=True).item()
    return {}

# Función para guardar embeddings actualizados
def guardar_embeddings(output_path, embeddings_dict):
    np.save(output_path, embeddings_dict)

# Función para extraer texto de un PDF
def pdf_to_text(pdf_path):
    images = convert_from_path(pdf_path, dpi=100)
    text = ''
    for image in images:
        text += pytesseract.image_to_string(image)
    return text

# Función para obtener embeddings de texto
def obtener_embeddings(texto, modelo, tokenizador):
    inputs = tokenizador(texto, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = modelo(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

# Función principal para actualizar embeddings
def actualizar_embeddings(pdf_path, filename, output_path):
    embeddings_dict = cargar_embeddings(output_path)
    
    if filename in embeddings_dict:
        print(f"El embedding para {filename} ya existe. No se realizará ninguna actualización.")
        return
    
    text = pdf_to_text(pdf_path)
    
    if text.strip():
        embedding = obtener_embeddings(text, modelo, tokenizador)
        embeddings_dict[filename] = embedding
        guardar_embeddings(output_path, embeddings_dict)
        print(f"Embeddings actualizados y guardados para {filename}")
    else:
        print(f"Advertencia: No se pudo extraer texto del archivo {filename}")