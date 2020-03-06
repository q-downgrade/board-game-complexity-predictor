import pandas as pd
from flask import Flask, jsonify, request, render_template
import pickle
import numpy as np
import os


os.environ['THEANO_FLAGS'] = 'optimizer=None'
app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_ROOT, "./models/model.pkl") 

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
