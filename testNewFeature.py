import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np
import innvestigate

import matplotlib.pyplot as plt



(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

input_shape = (28, 28, 1)


x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

x_train = ( x_train - x_train.mean() )/ 255
x_test = ( x_test - x_train.mean() ) / 255 ###x_train!

y_train, y_test = map(keras.utils.to_categorical, (y_train, y_test))



model = keras.models.Sequential([
    #keras.layers.Input(shape=input_shape),
    keras.layers.Conv2D(16, (5, 5), input_shape=input_shape, use_bias=False, activation="linear"),
    keras.layers.AvgPool2D(),


    keras.layers.Conv2D(32, (5, 5), use_bias=False, activation="linear"),
    keras.layers.AvgPool2D(),

    keras.layers.Flatten(),
    keras.layers.Dense(240,  use_bias=False),
    keras.layers.Dense(120,  use_bias=False),
    keras.layers.Dense(10, use_bias=False, activation="softmax"),

])


model.compile(loss="categorical_crossentropy"
              , optimizer="adam", metrics=["accuracy"])

#model.load_weights("myModelWeights.h5")

model.summary()

model_wo_sm = innvestigate.utils.keras.graph.model_wo_softmax(model)

#########



# Creating an analyzer
LRP_analyser = innvestigate.analyzer.LRPZ(model=model_wo_sm)

analysis = LRP_analyser.analyze(X=x_test[:50], neuron_selection="max_activation", stop_mapping_at_layers=["dense"])[0]


plt.figure("Heatmap")
plt.imshow(analysis[0].squeeze(-1), cmap="jet")
plt.show()

im = LRP_analyser.getIntermediate(['conv2d_input', 'conv2d', 'average_pooling2d', 'conv2d_1', 'average_pooling2d_1', 'flatten', 'dense', 'dense_1', 'dense_3'])



#test_model = keras.models.Model(inputs=model_wo_sm.inputs, outputs=model_wo_sm.layers[1].output)
#LRP_analyser = innvestigate.analyzer.LRPZ(model=test_model)
#analysis = LRP_analyser.analyze(X=x_test[:50], neuron_selection="max_activation")[0]

#plt.figure("Heatmap Old")
#plt.imshow(analysis[0].squeeze(-1), cmap="jet")
#plt.show()
