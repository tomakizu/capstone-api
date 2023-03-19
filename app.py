from flask import Flask, request, jsonify
import os
import prediction

app = Flask(__name__)

@app.route('/', methods=['POST'])
def predict():
    return prediction.getPredictionResult(dataset=request.json['dataset'], timestamp=request.json['timestamp'])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8014, debug=True)