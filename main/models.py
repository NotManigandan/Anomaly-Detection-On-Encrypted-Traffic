import torch
import tenseal as ts

# Logistic Regression
class LR(torch.nn.Module):

    def __init__(self, n_features):
        super(LR, self).__init__()
        self.lr = torch.nn.Linear(n_features, 1)
        
    def forward(self, x):
        out = torch.sigmoid(self.lr(x))
        return out
    
class EncryptedLR:
    
    def __init__(self, torch_lr):
        self.weight = torch_lr.lr.weight.data.tolist()[0]
        self.bias = torch_lr.lr.bias.data.tolist()
        
    def forward(self, enc_x):
        enc_out = enc_x.dot(self.weight) + self.bias
        return enc_out
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def encrypt(self, context):
        self.weight = ts.ckks_vector(context, self.weight)
        self.bias = ts.ckks_vector(context, self.bias)
        
    def decrypt(self, context):
        self.weight = self.weight.decrypt()
        self.bias = self.bias.decrypt()

# ANN/MLP
class NN(torch.nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.layer_1 = torch.nn.Linear(in_features=n_features, out_features=16)
        self.layer_2 = torch.nn.Linear(in_features=16, out_features=8)
        self.layer_3 = torch.nn.Linear(in_features=8, out_features=1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.layer_1(x)
        x = x*x
        x = self.layer_2(x)
        x = x*x
        x = self.layer_3(x)
        return x
    
class EncryptedANN:
    
    def __init__(self, torch_ann):
        self.layer_1_weight = torch_ann.layer_1.weight.T.data.tolist()
        self.layer_1_bias = torch_ann.layer_1.bias.data.tolist()

        self.layer_2_weight = torch_ann.layer_2.weight.T.data.tolist()
        self.layer_2_bias = torch_ann.layer_2.bias.data.tolist()
        
        self.layer_3_weight = torch_ann.layer_3.weight.T.data.tolist()
        self.layer_3_bias = torch_ann.layer_3.bias.data.tolist()
        
    def forward(self, enc_x):
        enc_x = enc_x.mm(self.layer_1_weight) + self.layer_1_bias
        enc_x.square_()
        
        enc_x = enc_x.mm(self.layer_2_weight) + self.layer_2_bias
        enc_x.square_()
                
        enc_x = enc_x.mm(self.layer_3_weight) + self.layer_3_bias
        
        return enc_x
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def encrypt(self, context):
        self.layer_1_weight =  ts.ckks_vector(context, self.layer_1_weight)
        self.layer_1_bias = ts.ckks_vector(context, self.layer_1_bias)

        self.layer_2_weight = ts.ckks_vector(context, self.layer_2_weight)
        self.layer_2_bias = ts.ckks_vector(context, self.layer_2_bias)
        
        self.layer_3_weight = ts.ckks_vector(context, self.layer_3_weight)
        self.layer_3_bias = ts.ckks_vector(context, self.layer_3_bias)
        
    def decrypt(self, context):
        self.layer_1_weight =  self.layer_1_weight.decrypt()
        self.layer_1_bias = self.layer_1_bias.decrypt()

        self.layer_2_weight = self.layer_2_weight.decrypt()
        self.layer_2_bias = self.layer_2_bias.decrypt()
        
        self.layer_3_weight = self.layer_3_weight.decrypt()
        self.layer_3_bias = self.layer_3_bias.decrypt()

# SVM
class SVM(torch.nn.Module):
    def __init__(self, n_features):
        super(SVM, self).__init__()
        self.dense_layer_1 = torch.nn.Linear(n_features, 1)

    def forward(self, x):
        output = self.dense_layer_1(x)
        return output 

class EncryptedSVM:
    
    def __init__(self, torch_svm):
        self.weight = torch_svm.dense_layer_1.weight.data.tolist()[0]
        self.bias = torch_svm.dense_layer_1.bias.data.tolist()
        
    def forward(self, enc_x):
        enc_out = enc_x.dot(self.weight) + self.bias
        return enc_out
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def encrypt(self, context):
        self.weight = ts.ckks_vector(context, self.weight)
        self.bias = ts.ckks_vector(context, self.bias)
        
    def decrypt(self, context):
        self.weight = self.weight.decrypt()
        self.bias = self.bias.decrypt()

# CNN
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()   
        # One convolution  layer becuase TenSEAL restricts this because of the use of im2col operation
        self.conv_layer = torch.nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding=0, stride=1)
        self.dense_layer_1 = torch.nn.Linear(in_features=36, out_features=8)
        self.dense_layer_2 = torch.nn.Linear(in_features=8, out_features=1)

    def forward(self, x):
        x = self.conv_layer(x)
        x = x * x
        x = x.view(x.size(0), -1)
        x = self.dense_layer_1(x)
        x = x * x
        x = self.dense_layer_2(x)
        return x
    
class EncCNN:
    def __init__(self, torch_nn):
        self.conv_layer_weight = torch_nn.conv_layer.weight.data.view(torch_nn.conv_layer.out_channels, torch_nn.conv_layer.kernel_size[0],torch_nn.conv_layer.kernel_size[1]).tolist()
        self.conv_layer_bias = torch_nn.conv_layer.bias.data.tolist()
        
        self.dense_layer_1_weight = torch_nn.dense_layer_1.weight.T.data.tolist()
        self.dense_layer_1_bias = torch_nn.dense_layer_1.bias.data.tolist()
        
        self.dense_layer_2_weight = torch_nn.dense_layer_2.weight.T.data.tolist()
        self.dense_layer_2_bias = torch_nn.dense_layer_2.bias.data.tolist()
        
        
    def forward(self, enc_x, windows_nb):
        enc_kernel_bias = []
        
        for kernel, bias in zip(self.conv_layer_weight, self.conv_layer_bias):
            y = enc_x.conv2d_im2col(kernel, windows_nb) + bias
            enc_kernel_bias.append(y)
        enc_op = ts.CKKSVector.pack_vectors(enc_kernel_bias)
        enc_op.square_()
        
        enc_op = enc_op.mm(self.dense_layer_1_weight) + self.dense_layer_1_bias
        enc_op.square_()
        
        enc_op = enc_op.mm(self.dense_layer_2_weight) + self.dense_layer_2_bias
        
        return enc_op
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    


def encrypted_evaluation(model, enc_x, model_name, windows_nb = None):
    op_list = []
    for i in enc_x:
        if model_name == "CNN":
            op = model(i, windows_nb)
        else:
            op = model(i)
        op_list.append(op)
    return op_list
 