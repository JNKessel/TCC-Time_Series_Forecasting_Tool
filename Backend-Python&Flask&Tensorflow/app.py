from flask import Flask,jsonify,request,render_template
from flask_cors import CORS
from BaselineModel import BaselineModel
from LinearModel import LinearModel
from DenseModel import DenseModel
from MultiStepDenseModel import MultiStepDenseModel
from ConvModel import ConvModel
from RecurrentLSTMModel import RecurrentLSTMModel

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
  return render_template('index.html')

@app.route('/baseline', methods=['POST'])
def baseline_model():

  request_data = request.get_json()
  baselineModel = BaselineModel(request_data)

  response =  jsonify(baselineModel.predict())
  response.headers.add("Access-Control-Allow-Origin", "*")
  return response

@app.route('/linear', methods=['POST'])
def linear_model():

  request_data = request.get_json()
  linearModel = LinearModel(request_data)

  response =  jsonify(linearModel.predict())
  response.headers.add("Access-Control-Allow-Origin", "*")
  return response

@app.route('/dense', methods=['POST'])
def dense_model():

  request_data = request.get_json()
  denseModel = DenseModel(request_data)

  response =  jsonify(denseModel.predict())
  response.headers.add("Access-Control-Allow-Origin", "*")
  return response

@app.route('/multistepdense', methods=['POST'])
def multi_step_dense_model():

  request_data = request.get_json()
  multiStepDenseModel = MultiStepDenseModel(request_data)

  response =  jsonify(multiStepDenseModel.predict())
  response.headers.add("Access-Control-Allow-Origin", "*")
  return response

@app.route('/convolutional', methods=['POST'])
def conv_model():

  request_data = request.get_json()
  convModel = ConvModel(request_data)

  response =  jsonify(convModel.predict())
  response.headers.add("Access-Control-Allow-Origin", "*")
  return response

@app.route('/recurrentlstm', methods=['POST'])
def recurrent_lstm_model():

  request_data = request.get_json()
  recurrentLSTMModel = RecurrentLSTMModel(request_data)

  response =  jsonify(recurrentLSTMModel.predict())
  response.headers.add("Access-Control-Allow-Origin", "*")
  return response

app.run(port=5000)
