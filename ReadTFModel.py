#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-04-10 21:19:11
# @Author  : etwll (hietwll@gmail.com)
# @Link    : ${link}
# @Version : $Id$

import tensorflow as tf  
import numpy as np
import sklearn.preprocessing as pre
from sklearn.externals import joblib

modelbase = "../model-12-60-30-1"

#write net weights
dlayer = [12,60,30,1]
scopes = list(map(lambda x:'scope'+str(x),range(1,len(dlayer))))
print(scopes) 
reader = tf.train.NewCheckpointReader(modelbase+'/model.ckpt-30001')  
for i in range(0,len(dlayer)-1):
    scope = scopes[i]
    w = reader.get_tensor(scope+"/weights")
    b = reader.get_tensor(scope+"/biases")  
    print("scope is : %s"%scope) 
    print("w.shape:") 
    print(w.shape)
    print("b.shape:")
    print(b.shape)
    np.savetxt('weights'+str(i+1)+'.dat',w,delimiter="  ")
    np.savetxt('biases'+str(i+1)+'.dat',b,delimiter="  ")

#write transformer
scaler_input = joblib.load(modelbase+"/scalerx.save")
scaler_output = joblib.load(modelbase+"/scalery.save")
np.savetxt("scaler_input_mean.dat",scaler_input.mean_,delimiter="  ")
np.savetxt("scaler_input_var.dat",scaler_input.var_,delimiter="  ")
np.savetxt("scaler_output_mean.dat",scaler_output.mean_,delimiter="  ")
np.savetxt("scaler_output_var.dat",scaler_output.var_,delimiter="  ")

#test transformer
# testx = np.random.rand(1,12)
# testy = np.random.rand(1,1)
# print(scaler_input.transform(testx)-(testx-scaler_input.mean_)/scaler_input.var_**0.5)
# print(scaler_output.transform(testy)-(testy-scaler_output.mean_)/scaler_output.var_**0.5)
