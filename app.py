from flask import Flask, request, render_template
import pickle
import numpy as np
import os


app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))     # set file directory path

MODEL_PATH = os.path.join(APP_ROOT, "./models/model.pkl")  # set path to the model

model = pickle.load(open(MODEL_PATH, 'rb')) # load the pickled model



# routes
@app.route("/")                         # render the website
def index(): 
   return render_template('index.html') 

@app.route('/submit', methods=['GET', 'POST'])  # submit the form
def make_prediction():
    features = [int(x) for x in request.form.values()] # take the values from the form as a list
    final_features = [np.array(features)]       # convert the values into a numpy array
    prediction = model.predict(final_features)      # pass the array into the model for prediction
    return render_template('prediction.html', prediction = prediction[0])  # render the prediction page


if __name__ == '__main__':
    app.run(debug=True)
