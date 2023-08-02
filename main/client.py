import pandas as pd
import numpy as np
import subprocess
import time
import sys
import shutil
import os
import psutil
import tenseal as ts
import torch
import time
import pickle
import json
import base64
import requests
import warnings
warnings.filterwarnings("ignore")


def initiate_shell_script(send_ip, dst_ip, filename, interface):
    shell_script_path = "/home/kali/packetsniff.sh"
    # p = subprocess.Popen([shell_script_path, send_ip, dst_ip, filename, interface])
    p = subprocess.Popen(' '.join([shell_script_path, send_ip, dst_ip, filename, interface]), stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid)
    return p
    
def kill(proc_pid):
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()

def run_cicflowmeter(filename):
    cicflowmeter_path = "/home/kali/CICFlowMeter/jnetpcap/linux/jnetpcap-1.4.r1425/TCPDUMP_and_CICFlowMeter/CICFlowMeters/CICFlowMeter-4.0/bin/CICFlowMeter"
    subprocess.run([cicflowmeter_path])

def stop_process(p):
    p.terminate()
   
def create_context(N, q, scale, galois = False):
    poly_mod_degree = N
    coeff_mod_bit_sizes = q
    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, -1, coeff_mod_bit_sizes)
    ctx.global_scale = 2 ** scale
    if galois == True:
        ctx.generate_galois_keys()
    return ctx

def load_prepare_input(context, csv_path, model, n = None, m = None):
    df = pd.read_csv(csv_path)
    df = df.drop(df[df.Protocol  == 'Protocol'].index)
    cols = ['Flow Duration', 'Total Fwd Packet', 'Total Bwd packets',
            'Total Length of Fwd Packet', 'Total Length of Bwd Packet',
            'Fwd Packet Length Min', 'Fwd Packet Length Mean',
            'Fwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s',
            'Flow IAT Std', 'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Min',
            'Bwd IAT Min', 'Fwd PSH Flags', 'Fwd URG Flags', 'Bwd Packets/s',
            'Packet Length Max', 'FIN Flag Count', 'RST Flag Count',
            'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count', 'Down/Up Ratio',
            'FWD Init Win Bytes', 'Bwd Init Win Bytes',
            'Fwd Seg Size Min', 'Active Mean', 'Active Std', 'Label']
    org_cols_names = ['Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
       'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
       'Fwd Packet Length Min', 'Fwd Packet Length Mean',
       'Fwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s',
       'Flow IAT Std', 'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Min',
       'Bwd IAT Min', 'Fwd PSH Flags', 'Fwd URG Flags', 'Bwd Packets/s',
       'Max Packet Length', 'FIN Flag Count', 'RST Flag Count',
       'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count', 'Down/Up Ratio',
       'Init_Win_bytes_forward', 'Init_Win_bytes_backward',
       'min_seg_size_forward', 'Active Mean', 'Active Std', 'Label']
    df = df[cols]
    
    df.columns= org_cols_names
    x = df.drop('Label', axis=1)
    scaler_path = r"/home/kali/Desktop/main/models/scaler.pkl"
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    x = scaler.transform(x)
    
    #  If the model we are using is CNN, we have to reshape and perform im2col encoding
    windows_nb = 0
    if model == "CNN":
        x = x.reshape(len(df), n, m) # range(len(x))
        x = torch.from_numpy(x).float().unsqueeze(1)
        enc_x = []
        for i in range(10): # range(len(x))
            enc_x_instance, windows_nb = ts.im2col_encoding(context, x[i].view(n, m).tolist(), 3, 3, 1)
            enc_x.append(enc_x_instance)
    else:
        enc_x = []
        x = torch.from_numpy(x).float()
        for i in range(10): # range(len(x))
            enc_x.append(ts.ckks_vector(context, x[i].tolist()))
        

    return enc_x, windows_nb

def serialize_input(context, enc_x, windows_nb):
    server_context = context.copy()
    server_context.make_context_public()
    server_context = base64.b64encode(server_context.serialize()).decode()

    encrypted_input = []
    for i in range(len(enc_x)):
        serialized = base64.b64encode(enc_x[i].serialize()).decode()
        encrypted_input.append(serialized)
    windows_nb_bytes = windows_nb.to_bytes((windows_nb.bit_length() + 7) // 8, 'big')
    windows_nb_encoded = base64.b64encode(windows_nb_bytes).decode()    

    client_data = {
        "context" : server_context,
        "data" : encrypted_input,
        "window" : windows_nb_encoded
    }

    return json.dumps(client_data)

def prepare_tenseal_parameters(model):
    if MODEL == "SVM" or MODEL == "LR":
        poly_mod_degree = 8192
        coeff_mod_bit_sizes = [60, 40, 60]
        scl = 40
    elif MODEL == "ANN":
        poly_mod_degree = 16384
        coeff_mod_bit_sizes = [60, 40, 40, 40, 40, 40, 60]
        scl = 40
    elif MODEL == "CNN":
        poly_mod_degree = 16384
        coeff_mod_bit_sizes = [40, 31, 31, 31, 31, 31, 31, 40]
        scl = 31   
    
    return poly_mod_degree, coeff_mod_bit_sizes, scl


def main(MODEL, s_ip, d_ip, interface, time_to_sniff):  
    filename = "test.pcap"
    p = initiate_shell_script(s_ip, d_ip, filename, interface)
    time.sleep(time_to_sniff)  
    stop_process(p)
    kill(p.pid)
    if p.poll() is None:  
        print("The script didn't terminate as expected.")
    else:
    	print("Finished capturing the packets, proceeeding to extract features using CICFlowMeter.")
    shutil.move(filename, "/home/kali/Desktop/main/data/in/" + filename)
    time.sleep(10)  
    run_cicflowmeter(filename)
    
    
    poly_mod_degree, coeff_mod_bit_sizes, scl = prepare_tenseal_parameters(MODEL)
    # print(poly_mod_degree, coeff_mod_bit_sizes, scl)
    ctx = create_context(poly_mod_degree, coeff_mod_bit_sizes, scl, True)
    enc_ip, windows_nb = load_prepare_input(ctx, r"/home/kali/Desktop/main/data/out/test_ISCX.csv", MODEL, 6, 5)
    sending_data = serialize_input(ctx, enc_ip, windows_nb)
    response = requests.post("http://127.0.0.1:5000/compute", json = sending_data)
    # print(response)
    response_data = response.json()
    encrypted_output = []
    for enc in response_data['data']:
        enc = base64.b64decode(enc)
        encrypted_output.append(ts.ckks_vector_from(ctx, enc).decrypt())
    if MODEL == "SVM":
        op = torch.sign(torch.tensor(encrypted_output))
    else:
        op = torch.sigmoid(torch.tensor(encrypted_output))
    op = op.squeeze().numpy()
    op = op.astype(float)  # Ensure the array is of type float
    labels = np.where(op == 1, "Malicious", "Benign")  # Replace 1 with "Malicious" and other values with "Benign"
    print(labels)
    
if __name__ == "__main__":
    MODEL = "ANN"
    s_ip = "192.168.177.129"
    d_ip = "192.168.177.135"
    interface = "eth0"
    time_to_sniff = 1
    main(MODEL, s_ip, d_ip, interface, time_to_sniff)