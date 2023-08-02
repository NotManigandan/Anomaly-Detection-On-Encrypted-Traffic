import tenseal as ts
import numpy as np

def inverse(context, enc_vector, itr):
    a = 2 - enc_vector
    b = 1 - enc_vector
    for i in range(itr): 
        b = b.square_()
        a = a * (1 + b)
    return a


def comparision(context, enc_a, enc_b, itr_1, itr_2, t, m):
    ''' 
      d ---> itr_1
      d' ---> itr_2 
      m^t = k - k is the scaling factor of sigmoid function
    '''
    temp = inverse(context, (enc_a + enc_b)*0.5, itr_2) # 2*itr_2
    enc_a = enc_a*temp*0.5 # 3
    enc_b = 1 - enc_a
    for i in range(t):
        enc_a_m  = enc_a.pow_(m)
        enc_b_m = enc_b.pow_(m)
        # print(f"t = {i}")
        # print(enc_a_m.decrypt())
        # print(enc_b_m.decrypt())
        inv = inverse(context, enc_a_m + enc_b_m, itr_1)
        enc_a = enc_a_m*inv
        enc_b = 1 - enc_a
    return enc_a