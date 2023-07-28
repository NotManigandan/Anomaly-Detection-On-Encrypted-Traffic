from flask import Flask, request
import tenseal as ts
import models
from pathlib import Path
import json
import base64
import torch
import os

app = Flask(__name__)

def preapre_client_data(data):
    server_context = base64.b64decode(data["context"])
    context = ts.context_from(server_context)
    encrypted_input = []
    for enc in data['data']:
        enc = base64.b64decode(enc)
        encrypted_input.append(ts.ckks_vector_from(context, enc))
    windows_nb_bytes = base64.b64decode(data["window"])
    windows_nb = int.from_bytes(windows_nb_bytes, 'big')
    
    return context, encrypted_input, windows_nb

def load_model(name):
    MODEL_PATH = Path("models")
    if name == "CNN":
        MODEL_NAME = "cnn_3_layers.pth"
        MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
        model = models.CNN()
        model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
        enc_model = models.EncCNN(model)

    elif name == "ANN":
        MODEL_NAME = "ann_3_layers.pth"
        MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
        model = models.NN(30)
        model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
        enc_model = models.EncryptedANN(model)
    
    elif name == "LR":
        MODEL_NAME = "lr.pth"
        MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
        model = models.LR(30)
        model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
        enc_model = models.EncryptedLR(model)

    elif name =="SVM":
        MODEL_NAME = "svm.pth"
        MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
        model = models.SVM(30)
        model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
        enc_model = models.EncryptedSVM(model)
    
    return enc_model

@app.route('/compute', methods=['POST'])
def compute():
    MODEL = "SVM" 
    client_data = request.get_json()
    # if client_data is None:
    #     print('Failed to parse JSON')
    # else:
    #     print('Parsed JSON: ', client_data)
    if isinstance(client_data, str):  
        client_data = json.loads(client_data)  

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    context_no_sk, encrypted_input, windows_nb = preapre_client_data(client_data)

    enc_model = load_model(MODEL)

    server_output = models.encrypted_evaluation(enc_model, encrypted_input, MODEL, windows_nb)

    encrypted_output = []
    for i in range(len(server_output)):
        serialized = base64.b64encode(server_output[i].serialize()).decode()
        encrypted_output.append(serialized)
    server_response = {
        "data" : encrypted_output
    }

    return json.dumps(server_response)

if __name__ == '__main__':
    app.run(debug=True)