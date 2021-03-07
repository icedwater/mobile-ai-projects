#! /usr/bin/env python3

from flask import Flask, request
from keras.models import load_model
from keras import backend as K

import numpy
app = Flask(__name__)

@app.route('/')
def hello_world():
    return "Placeholder for Predictor."

@app.route("/predict", methods=["POST"])
def add():
    req_data = request.get_json()
    bizprop = req_data["bizprop"]
    rooms = req_data["rooms"]
    age = req_data["age"]
    highways = req_data["highways"]
    tax = req_data["tax"]
    ptratio = req_data["ptratio"]
    lstat = req_data["lstat"]
    std_simple_net = load_model("standardised_simple_shallow_seq_net.h5")
    value = std_simple_net.predict_on_batch(numpy.array([[bizprop, rooms, age, highways, tax, ptratio, lstat]], dtype=float))
    K.clear_session()
    return str(value)

