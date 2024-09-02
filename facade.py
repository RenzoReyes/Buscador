import numpy as np
import os
import subprocess
import json
from typing import List, Dict, Any
from crawler_daemon import start_daemon  # Importar la función para iniciar el daemon

class BuscadorFacade:
    def __init__(self, procesar_consulta_script: str, ranking_script: str, ruta_documentos: str):
        self.procesar_consulta_script = procesar_consulta_script
        self.ranking_script = ranking_script
        self.ruta_documentos = ruta_documentos

        # Ruta del índice invertido
        self.output_path = r'C:\Users\56974\Desktop\seminario 2024\codigo python github\indice_invertido_con_stopwords_normalizados_github.json'

        # Iniciar el daemon del crawler en segundo plano
        start_daemon(self.ruta_documentos, self.output_path)
        
        # Ruta del archivo embeddings.npy
        embeddings_path = r'C:\Users\56974\Desktop\seminario 2024\codigo python github\embeddings.npy'
        
        # Cargar los embeddings desde el archivo
        if os.path.exists(embeddings_path):
            try:
                self.embeddings = np.load(embeddings_path, allow_pickle=True).item()
            except Exception as e:
                raise RuntimeError(f"Error al cargar embeddings: {e}")
        else:
            raise FileNotFoundError(f"No se encontró el archivo embeddings.npy en la ruta {embeddings_path}")

    def buscar_documentos(self, query: str) -> List[Dict[str, Any]]:
        try:
            # Ejecutar el script de procesamiento de consulta con BERT
            consulta_resultado = subprocess.run(
                ['python', self.procesar_consulta_script, query],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"Consulta result: {consulta_resultado.stdout}")  # Depuración

            terminos_clave = json.loads(consulta_resultado.stdout)
            
            # Crear un archivo temporal con los términos clave
            with open('terminos_clave.json', 'w') as f:
                json.dump(terminos_clave, f)
            
            # Ejecutar el script de ranking con los términos clave
            ranking_resultado = subprocess.run(
                ['python', self.ranking_script, query, self.ruta_documentos],
                capture_output=True,
                text=True,
                check=True
            )
            
            print(f"Ranking result: {ranking_resultado.stdout}")  # Depuración
            resultados = json.loads(ranking_resultado.stdout)
            return resultados
        
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Error al ejecutar los scripts: {e.stderr}")
        except Exception as e:
            raise RuntimeError(f"Error al buscar documentos: {e}")
