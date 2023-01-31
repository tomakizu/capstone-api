from flask import Flask
import os
import prediction

app = Flask(__name__)

@app.route('/')
def predict():
    return prediction.getPredictionResult()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8014, debug=True)