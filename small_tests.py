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
    a = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
    outputs = tf.keras.layers.Dense(5, activation="softmax")(a)
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

def run_analysis_old(input, model, name, analyzer, neuron_selection):
    print("Old Test")
    print("Model Name: ", name)
    print("Analyzer Class: ", analyzer)
    print("Param neuron_selection: ", neuron_selection)

    if neuron_selection is None or isinstance(neuron_selection, str):
        if neuron_selection is None:
            neuron_selection = "all"
        model = innvestigate.utils.keras.graph.model_wo_softmax(model)
        a = time.time()
        ana = analyzer(model, neuron_selection_mode = neuron_selection)
        R = ana.analyze(input)
        b = time.time()
        print("Time Passed: ", b - a)
        print("explanation: ", np.shape(R))
        return R
    else:
        neuron_selection_mode = "index"
        model = innvestigate.utils.keras.graph.model_wo_softmax(model)
        a = time.time()
        ana = analyzer(model, neuron_selection_mode=neuron_selection_mode)
        R = ana.analyze(input, neuron_selection=neuron_selection)
        b = time.time()
        print("Time Passed: ", b - a)
        print("explanation: ", np.shape(R))
        return R

#----------------------------------------------------------------------------------
#Tests

model_cases = [
    #SimpleDense,
    MultiIn,
    #MultiConnect,
    #VGG16
]

analyzer_cases = [
    innvestigate.analyzer.ReverseAnalyzerBase,
    #innvestigate.analyzer.LRPSequentialPresetBFlat_new,
]

analyzer_cases_old = [
    innvestigate.analyzer.base.ReverseAnalyzerBase,
    #innvestigate.analyzer.LRPSequentialPresetBFlat,
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
            R_old = run_analysis_old(
                input=input,
                model=model,
                name=name,
                analyzer=analyzer_cases_old[a],
                neuron_selection=neuron_selection_case
            )
            print(np.all(R_new[0] == R_old))
            print(R_new)
            print(R_old)
            print("----------------------------------------------------------------------------------")
