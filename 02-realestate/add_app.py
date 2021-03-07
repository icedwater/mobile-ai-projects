#! /usr/bin/env python3

from flask import Flask, request
app = Flask(__name__)

@app.route('/')
def hello_world():
    return "POST me two numbers at /add, I'll add them for you."

@app.route("/add", methods=["POST"])
def add():
    try:
        req_data = request.get_json()
        number_1 = req_data["number_1"]
        number_2 = req_data["number_2"]
        answer = str(int(number_1) + int(number_2))
    except ValueError:
        try:
            answer = str(float(number_1) + float(number_2))
        except:
            answer = "Sorry, I didn't get two numbers."
    return answer

