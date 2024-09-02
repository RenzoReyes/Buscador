from flask import Flask, request, jsonify, render_template, send_file, abort
from facade import BuscadorFacade  # Asegúrate de que facade.py está en el mismo directorio
import os

app = Flask(__name__)

# Configuración de la fachada
RUTA_DOCUMENTOS = r"C:\Users\56974\Desktop\seminario 2024\codigo python github\decretos_2023_test"
PROCESAR_CONSULTA_SCRIPT = 'procesar_consulta.py'
RANKING_SCRIPT = 'ranking.py'

# Definir la instancia de la fachada
facade = BuscadorFacade(PROCESAR_CONSULTA_SCRIPT, RANKING_SCRIPT, RUTA_DOCUMENTOS)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/buscar', methods=['POST'])
def buscar():
    query = request.form.get('query')
    if not query:
        return render_template('error.html', error_message='Por favor, ingrese una consulta.')

    try:
        resultados = facade.buscar_documentos(query)
        if not resultados:
            return render_template('error.html', error_message='No se encontraron documentos para la consulta.')
        return render_template('resultados.html', resultados=resultados)
    except Exception as e:
        return render_template('error.html', error_message=f'Ocurrió un error: {e}')

@app.route('/ver/<doc_id>')
def ver_documento(doc_id):
    ruta_archivo = os.path.join(RUTA_DOCUMENTOS, f"{doc_id}.pdf")
    if os.path.exists(ruta_archivo):
        # Cambiamos para enviar el archivo como un archivo para que el navegador lo muestre o lo descargue
        return send_file(ruta_archivo, as_attachment=False)
    else:
        return render_template('error.html', error_message='Documento no encontrado.')

@app.route('/descargar/<doc_id>')
def descargar_documento(doc_id):
    ruta_archivo = os.path.join(RUTA_DOCUMENTOS, f"{doc_id}.pdf")
    if os.path.exists(ruta_archivo):
        return send_file(ruta_archivo, as_attachment=True)
    else:
        return render_template('error.html', error_message='Documento no encontrado.')

if __name__ == '__main__':
    app.run(debug=True)
