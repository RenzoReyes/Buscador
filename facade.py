from crawler import start_crawler, check_for_new_files
from actualizar_embeddings import cargar_embeddings, actualizar_embeddings
from actualizar_indice_invertido import cargar_indice, actualizar_indice_invertido
from transformers import BertTokenizer, BertModel
from pymongo import MongoClient
import numpy as np
import os
import torch


class BuscadorFacade:
    def __init__(self, ruta_documentos: str, debug: bool = False):
        self.ruta_documentos = ruta_documentos
        self.debug = debug

        # Rutas
        self.output_path = r'C:\Users\56974\Desktop\seminario 2024\codigos python avanzados\indice_invertido_con_stopwords_normalizados_avanzado_test.json'
        self.embeddings_path = r'C:\Users\56974\Desktop\seminario 2024\codigos python avanzados\embeddings_avanzado_test.npy'

        # Conexión a MongoDB
        client = MongoClient('mongodb://localhost:27017/')
        db = client['indice_invertido_decretos_munvalp_test']
        self.collection = db['indice_invertido_test']

        # Inicializar BERT
        self.tokenizador = BertTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased')
        self.modelo = BertModel.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased')

        # Cargar embeddings
        self.embeddings = cargar_embeddings(self.embeddings_path)

        # Iniciar el crawler
        start_crawler(self.ruta_documentos, self.output_path, self.embeddings_path)

    def buscar_documentos(self, query: str):
        """
        Buscar documentos relevantes basados en la consulta.
        """
        try:
            print(f"[DEBUG] Consulta recibida: {query}")
            resultados = self._buscar_por_terminos(query)
            if not resultados:
                print("[DEBUG] No se encontraron documentos por términos.")
                resultados = self._procesar_con_embeddings(query)
            return resultados
        except Exception as e:
            print(f"[ERROR] Error al buscar documentos: {e}")
            return []

    def _buscar_por_terminos(self, query: str):
        """
        Buscar documentos directamente por términos en el índice invertido.
        """
        try:
            terms = query.lower().split()
            doc_ids = []
            for term in terms:
                cursor = self.collection.find({"word": term})
                for resultado in cursor:
                    if "documents" in resultado:
                        doc_ids.extend([doc["documento"] for doc in resultado["documents"]])
            return list(set(doc_ids))
        except Exception as e:
            print(f"[ERROR] Error al buscar documentos por términos: {e}")
            return []

    def _procesar_con_embeddings(self, query: str):
        """
        Buscar documentos utilizando embeddings.
        """
        try:
            embedding_query = self.obtener_embeddings(query)
            resultados = []
            for doc_id, embedding_doc in self.embeddings.items():
                similitud = self.similitud_coseno(embedding_query, embedding_doc)
                resultados.append({"documento": doc_id, "similitud": similitud})
            return sorted(resultados, key=lambda x: x["similitud"], reverse=True)
        except Exception as e:
            print(f"[ERROR] Error al procesar con embeddings: {e}")
            return []

    def obtener_embeddings(self, texto: str):
        """
        Obtener embeddings para un texto usando BERT.
        """
        try:
            inputs = self.tokenizador(texto, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.modelo(**inputs)
            return outputs.last_hidden_state.mean(dim=1).numpy().squeeze()
        except Exception as e:
            print(f"[ERROR] Error al obtener embeddings: {e}")
            return None

    def similitud_coseno(self, vector1, vector2):
        """
        Calcular similitud del coseno entre dos vectores.
        """
        producto_punto = np.dot(vector1, vector2)
        magnitud1 = np.linalg.norm(vector1)
        magnitud2 = np.linalg.norm(vector2)
        if magnitud1 == 0 or magnitud2 == 0:
            return 0.0
        return producto_punto / (magnitud1 * magnitud2)

    def actualizar_indice(self, pdf_path: str):
        """
        Actualizar el índice invertido.
        """
        try:
            filename = os.path.basename(pdf_path)
            actualizar_indice_invertido(pdf_path, filename, self.output_path, self.collection)
            print(f"[INFO] Índice invertido actualizado para {filename}")
        except Exception as e:
            print(f"[ERROR] Error al actualizar el índice: {e}")

    def actualizar_embeddings(self, pdf_path: str):
        """
        Actualizar embeddings.
        """
        try:
            filename = os.path.basename(pdf_path)
            actualizar_embeddings(pdf_path, filename, self.embeddings_path)
            print(f"[INFO] Embeddings actualizados para {filename}")
        except Exception as e:
            print(f"[ERROR] Error al actualizar embeddings: {e}")

    def ejecutar_crawler(self):
        """
        Ejecutar el crawler para procesar nuevos archivos.
        """
        try:
            check_for_new_files(self.ruta_documentos, self.output_path, self.embeddings_path)
        except Exception as e:
            print(f"[ERROR] Error al ejecutar el crawler: {e}")
