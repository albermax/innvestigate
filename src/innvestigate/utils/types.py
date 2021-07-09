"""Custom types used in iNNvestigate"""

from typing import Any, Callable, List, Optional, Sequence, Tuple, TypeVar, Union

from keras import Model
from keras.layers import Layer
from tensorflow import Tensor

# from tensorflow.python.types import Tensor
from typing_extensions import TypedDict

T = TypeVar("T")  # Generic type, can be anything

# Keras commonly accepts and returns both tensors and lists of tensors.
# Since iNNvestigate builds on Keras layers, it does too.
# These generic type aliases can be used instead of things like
#   Union[Tensor, List[Tensor]]
OptionalList = Union[T, List[T]]
OptionalSequence = Union[T, Sequence[T]]

# Shapes of tensors are described using tuples of ints (and sometimes Nones).
ShapeTuple = Tuple[Optional[int], ...]

# Type for boolean checks on Keras layers
LayerCheck = Callable[[Layer], bool]

# Used for LRP rules
ReverseRule = Tuple[LayerCheck, Any]  # TODO: replace Any with ReverseMappingBase


class ModelCheckDict(TypedDict):
    """ "Adds type hints to model check dicts."""

    check: LayerCheck
    message: str
    check_type: str


class CondReverseMapping(TypedDict):
    """Adds type hints to conditional reverse mapping dicts."""

    condition: LayerCheck
    mapping: Callable  # TODO: specify type
    name: Optional[str]


class ReverseState(TypedDict):
    """Adds type hints to state used in analyzers of type ReverseAnalyzerBase."""

    layer: Layer
    model: Model
    nid: int
    stop_mapping_at_tensors: List[Tensor]


class NodeDict(TypedDict):
    """Adds type hints to NodeDicts.

    Contains the following items:
    * `nid`: the node id.
    * `layer`: the layer creating this node.
    * `Xs`: the input tensors (only valid if not in a nested container).
    * `Ys`: the output tensors (only valid if not in a nested container).
    * `Xs_nids`: the ids of the nodes creating the Xs.
    * `Ys_nids`: the ids of nodes using the according output tensor.
    * `Xs_layers`: the layer that created the according input tensor.
    * `Ys_layers`: the layers using the according output tensor.

    """

    nid: Optional[int]
    layer: Layer
    Xs: List[Tensor]
    Ys: List[Tensor]
    Xs_nids: List[Optional[int]]
    Ys_nids: List[Union[List[int], List[None]]]
    Xs_layers: List[Layer]
    Ys_layers: List[List[Layer]]


class ReverseTensorDict(TypedDict):

    id: Tuple[int, int]  # Node ID and order in reversed_tensor_list
    tensor: Optional[Tensor]  # reversed tensor
    tensors: Optional[List[Tensor]]
    final_tensor: Optional[Tensor]
