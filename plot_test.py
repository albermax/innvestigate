from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, MaxPooling2D
from tensorflow.keras.models import Sequential
import tensorflow as tf
import innvestigate
import tensorflow.keras as keras
import numpy as np
import time

import matplotlib.pyplot as plt

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

def LeNet():

    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    if keras.backend.image_data_format == "channels_first":
        input_shape = (1, 28, 28)
    else:
        input_shape = (28, 28, 1)

    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    x_train = (x_train - x_train.mean()) / 255
    x_test = (x_test - x_train.mean()) / 255  ###x_train!

    y_train, y_test = map(keras.utils.to_categorical, (y_train, y_test))

    model = keras.models.Sequential([
        keras.layers.Conv2D(16, (5, 5), input_shape=input_shape, use_bias=False, activation="linear"),
        keras.layers.MaxPooling2D(),

        keras.layers.Conv2D(32, (5, 5), use_bias=False, activation="linear"),
        keras.layers.MaxPooling2D(),

        keras.layers.Flatten(),
        keras.layers.Dense(240, use_bias=False),
        keras.layers.Dense(120, use_bias=False),
        keras.layers.Dense(10, use_bias=False, activation="softmax"),
        #keras.layers.Softmax()
    ])

    model.compile(loss="categorical_crossentropy"
                  , optimizer="adam", metrics=["accuracy"])

    #model.fit(x_train, y_train, epochs=2, batch_size=128, validation_data=(x_test, y_test))
    #model.save_weights('New_LeNetWeights.h5')

    model.load_weights('LeNetWeights.h5')

    return x_test, model, "LeNet"

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
    print("explanation: ", np.shape(np.array(R)))
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
        print("explanation: ", np.shape(np.array(R)))
        return R
    else:
        neuron_selection_mode = "index"
        model = innvestigate.utils.keras.graph.model_wo_softmax(model)
        a = time.time()
        ana = analyzer(model, neuron_selection_mode=neuron_selection_mode)
        R = ana.analyze(input, neuron_selection=neuron_selection)
        b = time.time()
        print("Time Passed: ", b - a)
        print("explanation: ", np.shape(np.array(R)))
        return R

#----------------------------------------------------------------------------------
#Tests

model_cases = [
    #SimpleDense,
    #MultiIn,
    #MultiConnect,
    #VGG16,
    LeNet
]

analyzer_cases = [
    #innvestigate.analyzer.ReverseAnalyzerBase,
    innvestigate.analyzer.LRPZ_new,
    #innvestigate.analyzer.LRPZIgnoreBias_new,
    #innvestigate.analyzer.LRPZPlus_new,
    #innvestigate.analyzer.LRPZPlusFast_new,
    #innvestigate.analyzer.LRPEpsilon_new,
    #innvestigate.analyzer.LRPEpsilonIgnoreBias_new,
    #innvestigate.analyzer.LRPWSquare_new,
    #innvestigate.analyzer.LRPFlat_new,
    #innvestigate.analyzer.LRPAlpha2Beta1_new,
    #innvestigate.analyzer.LRPAlpha2Beta1IgnoreBias_new,
    #innvestigate.analyzer.LRPAlpha1Beta0_new,
    #innvestigate.analyzer.LRPAlpha1Beta0IgnoreBias_new,
    #innvestigate.analyzer.LRPSequentialCompositeA_new,
    #innvestigate.analyzer.LRPSequentialCompositeB_new,
    #innvestigate.analyzer.LRPSequentialCompositeAFlat_new,
    #innvestigate.analyzer.LRPSequentialCompositeBFlat_new,
]

analyzer_cases_old = [
    #innvestigate.analyzer.base.ReverseAnalyzerBase,
    innvestigate.analyzer.LRPZ,
    #innvestigate.analyzer.LRPZIgnoreBias,
    #innvestigate.analyzer.LRPZPlus,
    #innvestigate.analyzer.LRPZPlusFast,
    #innvestigate.analyzer.LRPEpsilon,
    #innvestigate.analyzer.LRPEpsilonIgnoreBias,
    #innvestigate.analyzer.LRPWSquare,
    #innvestigate.analyzer.LRPFlat,
    #innvestigate.analyzer.LRPAlpha2Beta1,
    #innvestigate.analyzer.LRPAlpha2Beta1IgnoreBias,
    #innvestigate.analyzer.LRPAlpha1Beta0,
    #innvestigate.analyzer.LRPAlpha1Beta0IgnoreBias,
    #innvestigate.analyzer.LRPSequentialPresetA,
    #innvestigate.analyzer.LRPSequentialPresetB,
    #innvestigate.analyzer.LRPSequentialPresetAFlat,
    #innvestigate.analyzer.LRPSequentialPresetBFlat,
]

neuron_selection_cases = [
    None,
    "max_activation",
    "all",
    0,
    #[0, 1, 2],
    #np.array([0, 1, 2])
]

test_results = []
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

            plt.figure(f"{name} {analyzer_case} {neuron_selection_case}")
            image_index = 0
            plt.subplot(3,1, 1)
            plt.imshow(input[image_index].squeeze(-1))
            plt.subplot(3, 1, 2)
            plt.title("R_new")
            plt.imshow(R_new[0][image_index].squeeze(-1), cmap="jet")
            plt.subplot(3, 1, 3)
            plt.title("R_old")
            plt.imshow(R_old[image_index].squeeze(-1), cmap="jet")
            plt.show()


            if len(R_new) > 1:
                comp = np.sum([r_n - r_o for r_n, r_o in zip(R_new, R_old)]) < 0.0001
            else:
                comp = np.sum(R_new[0] - np.array(R_old)) < 0.0001

            test_results.append(str(name) + " " + str(analyzer_case).split("'")[-2].split(".")[-1] + " " + str(
                neuron_selection_case) + ": " + str(comp))

            print(comp)
            print(R_new)
            print(R_old)
            print("----------------------------------------------------------------------------------")

print("----------------------------------------------------------------------------------")
print("----------------------------------------------------------------------------------")
print("--------------------------------------SUMMARY-------------------------------------")
print("----------------------------------------------------------------------------------")
print("----------------------------------------------------------------------------------")
for result in test_results:
    print(result)