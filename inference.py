"""
Load a model to predict user class
"""
import numpy as np
import pickle
from flask import Flask
from flask import request
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)


@app.route('/predict_class')
def predict_class():
    clf = read_model('class_model.pkl')
    age = request.args.get('age')
    is_male = request.args.get('is_male')
    height_cm = request.args.get('height_cm')
    weight_kg = request.args.get('weight_kg')
    # body_fat = request.args.get('body_fat')  ## name reformating
    sit_ups_counts = request.args.get('sit_ups_counts')
    # broad_jump_cm = request.args.get('broad_jump_cm')

    d = {'age': [age], 'is_male': [is_male], 'height_cm': [height_cm], 'weight_kg': [weight_kg],
         'sit_ups_counts': [sit_ups_counts]}
    X = pd.DataFrame(d)
    y = clf.predict(X)

    return str(y[0])


def read_model(filename):
    with open(filename, 'rb') as f:
        lr = pickle.load(f)
    return lr


def main():
    app.run(host='0.0.0.0', port=8080)


if __name__ == '__main__':
    main()
