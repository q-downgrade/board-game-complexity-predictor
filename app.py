import pandas as pd
from flask import Flask, jsonify, request, render_template
import pickle
from sklearn.externals import joblib
import numpy as np
import os


# app
app = Flask(__name__)

model = None

MODEL_PATH = "./static/model.pkl"

def load_model():
    global model
    model = pickle.load(open(MODEL_PATH, 'rb'))


# routes
@app.route("/")
def index():
   return render_template('index.html')

@app.route('/submit', methods=['GET', 'POST'])
def make_prediction():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    return render_template('prediction.html', prediction = prediction)


if __name__ == '__main__':
    app.run(debug=True)
