import os
from flask import Flask, request, jsonify
import tensorflow as tf
from keras.models import load_model
import pickle

# Crear la aplicacion Flask
app = Flask(__name__)

# Cargar el modelo y el vectorizador
model = load_model('model.h5')
with open('vectorizer.pkl', 'rb') as f:
    vect = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    plot_title_year = data['plot_title_year']
    
    # Preprocesamiento
    X = vect.transform([plot_title_year])
    
    # Prediccion
    prediction = model.predict(X)
    
    # Crear respuesta
    response = {
        'predictions': prediction.tolist()
    }
    return jsonify(response)

# Bloque principal para iniciar el servidor Flask
if __name__ == '__main__':
    # Iniciar el servidor Flask
    app.run(debug=True, host='0.0.0.0', port=5001)
