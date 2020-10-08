from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, MaxPooling2D
from tensorflow.keras.models import Sequential
import tensorflow as tf
import innvestigate
import tensorflow.keras as keras
import numpy as np
import time

import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

import innvestigate

# (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
#
# train_images, test_images = train_images / 255.0, test_images / 255.0
#
# class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
#                'dog', 'frog', 'horse', 'ship', 'truck']
#
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(10))
#
# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])
#
# #history = model.fit(train_images, train_labels, epochs=1,
# #                    validation_data=(test_images, test_labels))
#
# analyser = innvestigate.create_analyzer("lrp.epsilon", model)
#
# analysis = analyser.analyze(test_images[0])
# print(analysis)


#---------------------------------------------------------------------------------------
#Models

def SimpleDense():
    inputs = tf.keras.Input(shape=(1,))
    #a = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
    outputs = tf.keras.layers.Dense(3, activation="softmax")(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    inp = np.random.rand(3, 1)

    return inp, model, "SimpleDense"

def Concat():
    inputs1 = tf.keras.Input(shape=(1,))
    inputs2 = tf.keras.Input(shape=(1,))
    z = tf.keras.layers.Concatenate()([inputs1, inputs2])
    outputs = tf.keras.layers.Dense(5, activation="softmax")(z)
    model = tf.keras.Model(inputs=[inputs1, inputs2], outputs=outputs)
    inp = [np.random.rand(3, 1), np.random.rand(3, 1)]

    return inp, model, "Concat"

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

def MultiAdd():
    inputs = tf.keras.Input(shape=(10,))
    x1 = keras.layers.Dense(120, use_bias=False)(inputs)
    x2 = keras.layers.Dense(190, use_bias=False)(inputs)
    x = keras.layers.Concatenate(axis=-1)([x1, x2])
    x = keras.layers.Dense(240, use_bias=False)(x)
    x1 = keras.layers.Dense(120, use_bias=False)(x)
    x2 = keras.layers.Dense(120, use_bias=False)(x)
    x = keras.layers.Add()([x1, x2])
    x = keras.layers.Dense(10, use_bias=False, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    inp = np.random.rand(3, 10)

    return inp, model, "MultiAdd"

def ConcatModel():
    inputs = keras.layers.Input(shape=(224, 224, 3))
    x = keras.layers.Conv2D(16, (5, 5), use_bias=False, activation="linear")(inputs)
    x = keras.layers.AvgPool2D()(x)
    x = keras.layers.Conv2D(32, (5, 5), use_bias=False, activation="linear")(x)
    x = keras.layers.AvgPool2D()(x)
    x = keras.layers.Flatten()(x)
    x1 = keras.layers.Dense(120, use_bias=False)(x)
    x2 = keras.layers.Dense(120, use_bias=False)(x)
    x = keras.layers.Concatenate()([x1, x2])

    x = keras.layers.Dense(10, use_bias=False, activation="softmax")(x)

    model = keras.models.Model(inputs=inputs, outputs=x)
    inp = np.random.rand(3, 224, 224, 3)

    return inp, model, "ConcatModel"

def VGG16():
    model = tf.keras.applications.VGG16(
        include_top=True,
        weights="imagenet",
    )
    loader = tf.keras.preprocessing.image_dataset_from_directory("/media/weber/f3ed2aae-a7bf-4a55-b50d-ea8fb534f1f5/Datasets/Imagenet/train/",
                                                                 batch_size=10,
                                                                 image_size=(224, 224), shuffle=False)
    for (data, label) in loader:
        inp = data
        break

    return inp, model, "VGG16"

def VGG16_modified():
    keras_model = tf.keras.applications.VGG16(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3)
    )
    inp = np.random.rand(3, 224, 224, 3)
    last = keras_model.output

    x = Flatten()(last)
    x = Dropout(0.5)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(200, activation='softmax')(x)

    model = tf.keras.models.Model(keras_model.input, x)

    return inp, model, "VGG16"

def Resnet50():
    pass

def Densenet():
    pass

#------------------------------------------------------------------------------------
#Analysis

def gregoire_black_firered(R, normalize=True):
    if normalize:
        R /= np.max(np.abs(R))
    x = R

    hrp  = np.clip(x-0.00,0,0.25)/0.25
    hgp = np.clip(x-0.25,0,0.25)/0.25
    hbp = np.clip(x-0.50,0,0.50)/0.50

    hbn = np.clip(-x-0.00,0,0.25)/0.25
    hgn = np.clip(-x-0.25,0,0.25)/0.25
    hrn = np.clip(-x-0.50,0,0.50)/0.50

    return np.concatenate([(hrp+hrn)[...,None],(hgp+hgn)[...,None],(hbp+hbn)[...,None]],axis = 2)

def run_analysis(input, model, name, analyzer, neuron_selection):
    print("New Test")
    print("Model Name: ", name)
    print("Analyzer Class: ", analyzer)
    print("Param neuron_selection: ", neuron_selection)
    model = innvestigate.utils.keras.graph.model_wo_softmax(model)
    model.summary()
    for i in range(3):
        a = time.time()
        ana = analyzer(model)
        R = ana.analyze(input, neuron_selection=neuron_selection)
        b = time.time()
        print("Iteration ", i, "Time Passed: ", b - a)
    return R

#----------------------------------------------------------------------------------
#Tests

model_cases = [
    #SimpleDense,
    #Concat,
    #MultiIn,
    #MultiConnect,
    #MultiAdd,
    #ConcatModel,
    VGG16,
    #VGG16_modified
]

analyzer_cases = [
    #innvestigate.analyzer.ReverseAnalyzerBase,
    #innvestigate.analyzer.LRPZ,
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
    #innvestigate.analyzer.LRPSequentialCompositeA,
    #innvestigate.analyzer.LRPSequentialCompositeB,
    #innvestigate.analyzer.LRPSequentialCompositeAFlat,
    #innvestigate.analyzer.LRPSequentialCompositeBFlat,
    #innvestigate.analyzer.LRPGamma,
    #innvestigate.analyzer.LRPRuleUntilIndex,
    innvestigate.analyzer.Gradient,
    innvestigate.analyzer.InputTimesGradient,
    innvestigate.analyzer.GuidedBackprop,
    innvestigate.analyzer.Deconvnet,
    innvestigate.analyzer.SmoothGrad,
    innvestigate.analyzer.IntegratedGradients,
]

neuron_selection_cases = [
    None,
    #"max_activation",
    #"all",
    #0,
    #[0, 1, 2],
    #np.array([0, 1, 2])
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
            #print(R_new)

            #if name == "VGG16":
            #    for key in R_new.keys():
            #        plt.figure("Heatmap")
            #        plt.title(str(name) + " " + str(analyzer_case) + " " + str(neuron_selection_case) + " " + str(key))
            #        img = gregoire_black_firered(np.mean(R_new  [key][0], axis=-1))
            #        plt.imshow(img)
            #        plt.show()
            print("----------------------------------------------------------------------------------")

print("----------------------------------------------------------------------------------")
print("----------------------------------------------------------------------------------")
print("--------------------------------------SUMMARY-------------------------------------")
print("----------------------------------------------------------------------------------")
print("----------------------------------------------------------------------------------")
