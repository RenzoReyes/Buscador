from pymongo import MongoClient

# Configuración de MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['indice_invertido_decretos_munvalp_test']
collection = db['indice_invertido_test']

# Validación
#print(f"[DEBUG] Tipo de 'collection' en config_db: {type(collection)}")