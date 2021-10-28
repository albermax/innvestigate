"""
Basic analysis of a random model to act as an entry point for debugging.
"""

import numpy as np
import tensorflow.keras.backend as kbackend
import tensorflow.keras.layers as klayers
import tensorflow.keras.models as kmodels

import innvestigate
import innvestigate.utils as iutils

# Create dummy input data
if kbackend.image_data_format == "channels_first":
    input_shape = (1, 28, 28)
else:
    input_shape = (28, 28, 1)


batch_size = 123
x = np.random.rand(batch_size, *input_shape).astype(np.float32)

# Create model
model = kmodels.Sequential(
    [
        klayers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        klayers.Conv2D(64, (3, 3), activation="relu"),
        klayers.MaxPooling2D((2, 2)),
        klayers.Flatten(),
        klayers.Dense(512, activation="relu"),
        klayers.Dense(10, activation="softmax"),
    ]
)

# Create model without trailing softmax
model = iutils.keras.graph.model_wo_softmax(model)

# check model functionality
y = model(x)
print(y)
print(y.shape)

# Create patterns in case PatternNet or Co. are used
patterns = [x for x in model.get_weights() if len(x.shape) > 1]

# Create analyze

# analyzer = innvestigate.create_analyzer("integrated_gradients", model)
# analyzer = innvestigate.create_analyzer("integrated_gradients", model, neuron_selection_mode="all")
# analyzer = innvestigate.create_analyzer("smoothgrad", model)
# analyzer = innvestigate.create_analyzer("smoothgrad", model, neuron_selection_mode="all")
# analyzer = innvestigate.create_analyzer("gradient", model)
# analyzer = innvestigate.create_analyzer("gradient", model, neuron_selection_mode="all")
# analyzer = innvestigate.create_analyzer("lrp.z", model)
# analyzer = innvestigate.create_analyzer("lrp.z", model, neuron_selection_mode="all")
# analyzer = innvestigate.create_analyzer("pattern.net", model, patterns=patterns)
analyzer = innvestigate.create_analyzer(
    "pattern.attribution", model, pattern_type="relu"
)
analyzer.fit(x, batch_size=1, verbose=0)
patterns = analyzer._patterns


a = analyzer.analyze(x)
# print(a)
print(f"analysis shape: {np.shape(a)}")
