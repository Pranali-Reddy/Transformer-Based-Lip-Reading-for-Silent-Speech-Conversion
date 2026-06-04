import tensorflow as tf
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.TimeDistributed(layers.Conv2D(32,(3,3),activation='relu'),
    input_shape=(20,64,64,3)),
    layers.TimeDistributed(layers.MaxPooling2D()),
    layers.TimeDistributed(layers.Flatten()),
    layers.LSTM(64),
    layers.Dense(32,activation='relu'),
    layers.Dense(4,activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy')
model.save("model/lip_model.h5")
