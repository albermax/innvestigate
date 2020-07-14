from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, MaxPooling2D
from tensorflow.keras.models import Sequential
import tensorflow as tf
import innvestigate
import tensorflow.keras as keras
import numpy as np
import time

"""
# Build model

x = tf.constant([[1], [2]], dtype=tf.float16)

#@tf.function
def f(x):
    model = Dense(10)
    ret = model(x)
    return ret

with tf.GradientTape() as tape:
    tape.watch(x)
    # Make prediction
    ret = f(x)

# Calculate gradients
model_gradients = tape.gradient(ret, x)
print(model_gradients)
"""
"""
inputs = tf.keras.Input(shape=(1,))
a = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(a)
y = tf.keras.layers.Dense(4, activation=tf.nn.relu)(a)
z = tf.keras.layers.Concatenate()([x, y])
outputs = tf.keras.layers.Dense(5, activation="softmax")(z)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
#"""
#"""
model = tf.keras.applications.VGG16(
    include_top=True,
    weights="imagenet",
)
#"""
inp = np.random.rand(2, 224, 224, 3)
#inp = np.random.rand(3, 1)
model = innvestigate.utils.keras.graph.model_wo_softmax(model)

a = time.time()
ana = innvestigate.analyzer.LRPAlpha2Beta1_new(model)
R = ana.analyze(inp, neuron_selection="max_activation")
b = time.time()
print(b-a)

a = time.time()
ana = innvestigate.analyzer.LRPAlpha2Beta1(model, neuron_selection_mode="max_activation")
R2 = ana.analyze(inp, neuron_selection=None)
b = time.time()
print(b-a)

import numpy as np
print(np.sum(R2-R))