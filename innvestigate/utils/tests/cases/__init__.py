
# Import all test cases
from .trivia import dot
from .trivia import skip_connection

from .mlp import mlp2
from .mlp import mlp3

from .cnn import cnn_1dim_c1_d1
from .cnn import cnn_1dim_c2_d1
from .cnn import cnn_2dim_c1_d1
from .cnn import cnn_2dim_c2_d1
from .cnn import cnn_3dim_c1_d1
from .cnn import cnn_3dim_c2_d1


# Convenience lists of test cases.
FAST = [
    "dot",
    "skip_connection",

    "mlp2",

    "cnn_2dim_c1_d1",
    "cnn_2dim_c2_d1",
]

PRECOMMIT = [
    "mlp3",

    "cnn_1dim_c1_d1",
    "cnn_1dim_c2_d1",
    "cnn_3dim_c1_d1",
    "cnn_3dim_c2_d1",
]
