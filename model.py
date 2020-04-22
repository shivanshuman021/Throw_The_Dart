from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

def get_model():
    model = Sequential()
    model.add(Dense(128,input_shape=(2,),activation = 'relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(32,activation='relu'))
    model.add(Dense(4,activation='softmax'))

    return model
