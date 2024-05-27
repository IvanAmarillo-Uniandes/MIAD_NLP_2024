{\rtf1\ansi\ansicpg1252\cocoartf2639
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import os\
from flask import Flask, request, jsonify\
import tensorflow as tf\
from keras.models import load_model\
import pickle\
\
# Crear la aplicaci\'f3n Flask\
app = Flask(__name__)\
\
# Cargar el modelo y el vectorizador\
model = load_model('model.h5')\
with open('vectorizer.pkl', 'rb') as f:\
    vect = pickle.load(f)\
\
@app.route('/predict', methods=['POST'])\
def predict():\
    data = request.json\
    plot_title_year = data['plot_title_year']\
    \
    # Preprocesamiento\
    X = vect.transform([plot_title_year])\
    \
    # Predicci\'f3n\
    prediction = model.predict(X)\
    \
    # Crear respuesta\
    response = \{\
        'predictions': prediction.tolist()\
    \}\
    return jsonify(response)\
\
# Bloque principal para iniciar el servidor Flask\
if __name__ == '__main__':\
    # Iniciar el servidor Flask\
    app.run(debug=True, host='0.0.0.0', port=5001)\
\
}