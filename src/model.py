from tensorflow.keras import Model
from tensorflow.keras.layers import *

def dnn_model(input_shape):
    
    x_input = Input(shape=input_shape)
    
    x1 = Bidirectional(LSTM(units=768, return_sequences=True))(x_input)
    x2 = Bidirectional(LSTM(units=512, return_sequences=True))(x1)
    x3 = Bidirectional(LSTM(units=256, return_sequences=True))(x2)
    
    z2 = Bidirectional(GRU(units=256, return_sequences=True))(x2)
    z3 = Bidirectional(GRU(units=128, return_sequences=True))(Add()([x3, z2]))
    
    x = Concatenate(axis=2)([x3, z2, z3])
    x = Bidirectional(LSTM(units=192, return_sequences=True))(x)
    
    x = Dense(units=128, activation='selu')(x)
    
    x_output = Dense(units=1)(x)

    model = Model(inputs=x_input, outputs=x_output, 
                  name='DNN_Model')
    return model
