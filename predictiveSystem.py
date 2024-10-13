# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle
#loading the saved model
loaded_model=pickle.load(open('/Users/aditisingh/Downloads/diabetes_trained_model.sav','rb'))
scaler2=pickle.load(open('/Users/aditisingh/Downloads/scaler2.sav','rb'))

input_data = (1, 103, 30, 38, 83, 43.3, 0.183, 33)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1) 
std_data = scaler2.transform(input_data_reshaped)
                                                               
prediction = loaded_model.predict(std_data)
print(prediction) # it returns a list
if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')