import math
import numpy as np
import torch
import json

from datetime import datetime
from argparse import ArgumentParser

import constant

parser = ArgumentParser()
parser.add_argument('--network' , type=str, default='pems'               , help='network name (PEMS/METR)')
parser.add_argument('--datetime', type=str, default='2017-01-01 00:00:00', help='current datetime'        )
args = parser.parse_args()

def datetime2timestamp(datetime_string):
    datetime_object = datetime.strptime(datetime_string, '%Y-%m-%d %H:%M:%S')
    return int(datetime_object.timestamp())

def get_dataset_index(dataset_name, timestamp):
    if dataset_name == 'pems':
        dataset_start_timestamp = constant.PEMS_START_TIMESTAMP
    elif dataset_name == 'metr':
        dataset_start_timestamp = constant.METR_START_TIMESTAMP
    return int((timestamp - dataset_start_timestamp) / 300)

def seq2instance(data, P, Q, isPredict=False):
    num_step, dims = data.shape
    num_sample = 1 if isPredict else num_step - P - Q + 1
    x = np.zeros(shape = (num_sample, P, dims))
    y = np.zeros(shape = (num_sample, Q, dims))
    for i in range(num_sample):
        x[i] = data[i : i + P]
        y[i] = data[i if isPredict else i + P: i + P + Q]
    return x, y

def process_traffic_data(traffic_file, dataset_index):
    traffic_data = np.squeeze(np.load(traffic_file)['data'][:, :, 0])

    num_step = traffic_data.shape[0]
    train_steps = round(num_step * 0.6)

    trainX, trainY = seq2instance(traffic_data[:train_steps], 12, 12)
    validateX, validateY = seq2instance(traffic_data[dataset_index : dataset_index + 1], 12, 12, True)

    mean, std = np.mean(trainX), np.std(trainX)
    validateX = (validateX - mean) / std

    return validateX, validateY, mean, std

def predict(model, validateX, validateY, mean, std):
    model.eval()
    num_val = validateX.shape[0]
    prediction = []
    label = []
    num_batch = math.ceil(num_val / 16)
    with torch.no_grad():
        for batch_index in range(num_batch):
            if isinstance(model, torch.nn.Module):
                start_index = batch_index * 16
                end_index   = min((batch_index + 1) * 16, num_val)

                X = torch.from_numpy(validateX[start_index : end_index]).float().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                Y = validateY[start_index : end_index]

                resultY = model(X)

                prediction.append(resultY.cpu().numpy() * std + mean)
                label.append(Y)
    
    prediction = np.concatenate(prediction, axis = 0)
    label = np.concatenate(label, axis = 0)

    return prediction, label

def getPredictionResult(dataset='pems', timestamp='2017-01-01 00:00:00'):
    pems_parameters = {'model': 'PEMS_bay', 'traffic_file': 'PEMS_bay.npz', 'SE_file': 'PEMS_bay.npy'}
    metr_parameters = {'model': 'METR_LA' , 'traffic_file': 'METR_LA.npz' , 'SE_file': 'METR_LA.npy' }

    timestamp     = datetime2timestamp(timestamp)
    dataset_index = get_dataset_index(dataset, timestamp)
    # load data
    model = torch.load('data/' + locals()[dataset + '_parameters'].get('model'))
    validateX, validateY, mean, std = process_traffic_data('data/' + locals()[dataset + '_parameters'].get('traffic_file'), dataset_index)
    SE = torch.from_numpy(np.load('data/' + locals()[dataset + '_parameters'].get('SE_file')).astype(np.float32)).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Prediction
    prediction, label = predict(model, validateX, validateY, mean, std)
    return json.dumps({'prediction': np.array2string(prediction[0][0]), 'actual' : np.array2string(label[0][0])})

# # process and validate dataset
# dataset = args.network.lower()
# if dataset != 'pems' and dataset != 'metr':
#     print('Invalid dataset name:', dataset)
#     exit()

# # process and validate datetime
# datetime_string = args.datetime
# try:
#     timestamp = datetime2timestamp(datetime_string)
#     if dataset == 'metr':
#         if timestamp < constant.METR_START_TIMESTAMP or timestamp >= constant.METR_END_TIMESTAMP:
#             print('Datetime out of range:', datetime_string, 'expected: 2012-03-01 00:00:00 - 2012-06-27 23:59:59')
#             exit()
#     elif dataset == 'pems':
#         if timestamp < constant.PEMS_START_TIMESTAMP or timestamp >= constant.PEMS_END_TIMESTAMP:
#             print('Datetime out of range:', datetime_string, 'expected: 2017-01-01 00:00:00 - 2017-06-30 22:59:59')
#             exit()
# except ValueError:
#     print('Invalid format:', datetime_string)
#     exit()

# timestamp     = datetime2timestamp(datetime_string)
# dataset_index = get_dataset_index(dataset, timestamp)

# # load data
# model = torch.load('data/' + locals()[dataset + '_parameters'].get('model'))
# validateX, validateY, mean, std = process_traffic_data('data/' + locals()[dataset + '_parameters'].get('traffic_file'), dataset_index)
# SE = torch.from_numpy(np.load('data/' + locals()[dataset + '_parameters'].get('SE_file')).astype(np.float32)).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# # Prediction
# prediction, label = predict(model, validateX, validateY, mean, std)
# print(prediction[0][0])
# print(label[0][0])