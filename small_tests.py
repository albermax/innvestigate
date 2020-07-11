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
x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
y = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
z = tf.keras.layers.Concatenate()([x, y])
outputs = tf.keras.layers.Dense(5, activation=tf.nn.relu)(z)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
"""
#"""
model = tf.keras.applications.VGG16(
    include_top=True,
    weights="imagenet",
)
#"""
inp = np.random.rand(5, 224, 224, 3)
#inp = np.random.rand(3, 1)
model = innvestigate.utils.keras.graph.model_wo_softmax(model)

a = time.time()
ana = innvestigate.analyzer.new_base.ReverseAnalyzerBase(model)
R = ana.analyze(inp, neuron_selection=None)
b = time.time()
print(b-a)

a = time.time()
ana = innvestigate.analyzer.base.ReverseAnalyzerBase(model, neuron_selection_mode="all")
R = ana.analyze(inp, neuron_selection=None)
b = time.time()
print(b-a)
#print(np.shape(R))