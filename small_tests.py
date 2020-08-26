from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, MaxPooling2D
from tensorflow.keras.models import Sequential
import tensorflow as tf
import innvestigate
import tensorflow.keras as keras
import numpy as np
import time

#---------------------------------------------------------------------------------------
#Models

def SimpleDense():
    inputs = tf.keras.Input(shape=(1,))
    #a = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
    outputs = tf.keras.layers.Dense(3, activation="softmax")(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    inp = np.random.rand(3, 1)

    return inp, model, "SimpleDense"

def MultiIn():
    inputs1 = tf.keras.Input(shape=(1,))
    inputs2 = tf.keras.Input(shape=(1,))
    a = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs1)
    x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs2)
    z = tf.keras.layers.Concatenate()([a, x])
    outputs = tf.keras.layers.Dense(5, activation="softmax")(z)
    model = tf.keras.Model(inputs=[inputs1, inputs2], outputs=outputs)
    inp = [np.random.rand(3, 1), np.random.rand(3, 1)]

    return inp, model, "MultiIn"

def MultiConnect():
    inputs = tf.keras.Input(shape=(1,))
    a = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
    x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
    z = tf.keras.layers.Concatenate()([a, x])
    outputs = tf.keras.layers.Dense(5, activation="softmax")(z)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    inp = np.random.rand(3, 1)

    return inp, model, "MultiConnect"

def VGG16():
    model = tf.keras.applications.VGG16(
        include_top=True,
        weights="imagenet",
    )
    inp = np.random.rand(3, 224, 224, 3)

    model.summary()

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.MaxPooling2D):
            print(layer.name)
            print(layer.get_weights())

    exit()

    return inp, model, "VGG16"

def Resnet50():
    pass

def Densenet():
    pass

#------------------------------------------------------------------------------------
#Analysis

def run_analysis(input, model, name, analyzer, neuron_selection):
    print("New Test")
    print("Model Name: ", name)
    print("Analyzer Class: ", analyzer)
    print("Param neuron_selection: ", neuron_selection)
    model = innvestigate.utils.keras.graph.model_wo_softmax(model)
    a = time.time()
    ana = analyzer(model)
    R = ana.analyze(input, neuron_selection=neuron_selection)
    b = time.time()
    print("Time Passed: ", b - a)
    print("explanation: ", np.shape(R))
    return R

#----------------------------------------------------------------------------------
#Tests

model_cases = [
    #SimpleDense,
    #MultiIn,
    #MultiConnect,
    VGG16
]

analyzer_cases = [
    innvestigate.analyzer.ReverseAnalyzerBase,
    innvestigate.analyzer.LRPZ,
    innvestigate.analyzer.LRPZIgnoreBias,
    innvestigate.analyzer.LRPZPlus,
    innvestigate.analyzer.LRPZPlusFast,
    innvestigate.analyzer.LRPEpsilon,
    innvestigate.analyzer.LRPEpsilonIgnoreBias,
    innvestigate.analyzer.LRPWSquare,
    innvestigate.analyzer.LRPFlat,
    innvestigate.analyzer.LRPAlpha2Beta1,
    innvestigate.analyzer.LRPAlpha2Beta1IgnoreBias,
    innvestigate.analyzer.LRPAlpha1Beta0,
    innvestigate.analyzer.LRPAlpha1Beta0IgnoreBias,
    innvestigate.analyzer.LRPSequentialCompositeA,
    innvestigate.analyzer.LRPSequentialCompositeB,
    innvestigate.analyzer.LRPSequentialCompositeAFlat,
    innvestigate.analyzer.LRPSequentialCompositeBFlat,
    #innvestigate.analyzer.LRPGamma
]

neuron_selection_cases = [
    None,
    "max_activation",
    "all",
    0,
    [0, 1, 2],
    np.array([0, 1, 2])
]

for model_case in model_cases:
    for a, analyzer_case in enumerate(analyzer_cases):
        for neuron_selection_case in neuron_selection_cases:
            input, model, name = model_case()
            R_new = run_analysis(
                input=input,
                model=model,
                name=name,
                analyzer=analyzer_case,
                neuron_selection=neuron_selection_case
            )
            tf.keras.backend.clear_session()


            #print("Explanation Shape:", np.shape(R_new))
            print("----------------------------------------------------------------------------------")

print("----------------------------------------------------------------------------------")
print("----------------------------------------------------------------------------------")
print("--------------------------------------SUMMARY-------------------------------------")
print("----------------------------------------------------------------------------------")
print("----------------------------------------------------------------------------------")
