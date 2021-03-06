# pylint: disable=global-variable-not-assigned
import os
import sys
from torch import nn
from torch.nn import functional as F
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from geffnet import config
    # from geffnet.activations.activations_me import *
    # from geffnet.activations.activations_jit import *
    # from geffnet.activations.activations import *
    from geffnet.activations.activations import swish, Swish, sigmoid, Sigmoid,\
        mish, Mish, tanh, Tanh, hard_sigmoid, hard_swish, HardSigmoid, HardSwish
    from geffnet.activations.activations_jit import swish_jit, SwishJit, mish_jit, MishJit
    from geffnet.activations.activations_me import swish_me, mish_me, hard_swish_me,\
        hard_sigmoid_me, HardSigmoidMe, HardSwishMe, SwishMe, MishMe
except ImportError:
    print('fail to import zen_nas modules')

_ACT_FN_DEFAULT = dict(
    swish=swish,
    mish=mish,
    relu=F.relu,
    relu6=F.relu6,
    sigmoid=sigmoid,
    tanh=tanh,
    hard_sigmoid=hard_sigmoid,
    hard_swish=hard_swish,
)

_ACT_FN_JIT = dict(
    swish=swish_jit,
    mish=mish_jit,
)

_ACT_FN_ME = dict(
    swish=swish_me,
    mish=mish_me,
    hard_swish=hard_swish_me,
    hard_sigmoid_jit=hard_sigmoid_me,
)

_ACT_LAYER_DEFAULT = dict(
    swish=Swish,
    mish=Mish,
    relu=nn.ReLU,
    relu6=nn.ReLU6,
    sigmoid=Sigmoid,
    tanh=Tanh,
    hard_sigmoid=HardSigmoid,
    hard_swish=HardSwish,
)

_ACT_LAYER_JIT = dict(
    swish=SwishJit,
    mish=MishJit,
)

_ACT_LAYER_ME = dict(
    swish=SwishMe,
    mish=MishMe,
    hard_swish=HardSwishMe,
    hard_sigmoid=HardSigmoidMe
)

_OVERRIDE_FN = {}
_OVERRIDE_LAYER = {}


def add_override_act_fn(name, function):
    """add function to dict"""
    global _OVERRIDE_FN
    _OVERRIDE_FN[name] = function


def update_override_act_fn(overrides):
    """update function dict"""
    assert isinstance(overrides, dict)
    global _OVERRIDE_FN
    _OVERRIDE_FN.update(overrides)


# pylint: disable=global-statement
def clear_override_act_fn():
    global _OVERRIDE_FN
    _OVERRIDE_FN = {}


def add_override_act_layer(name, function):
    """add layer to dict"""
    _OVERRIDE_LAYER[name] = function


def update_override_act_layer(overrides):
    assert isinstance(overrides, dict)
    global _OVERRIDE_LAYER
    _OVERRIDE_LAYER.update(overrides)


# pylint: disable=global-statement
def clear_override_act_layer():
    global _OVERRIDE_LAYER
    _OVERRIDE_LAYER = {}


def get_act_fn(name='relu'):
    """ Activation Function Factory
    Fetching activation fns by name with this function allows export or torch script friendly
    functions to be returned dynamically based on current config.
    """
    if name in _OVERRIDE_FN:
        return _OVERRIDE_FN[name]
    no_me = config.is_exportable() or config.is_scriptable() or config.is_no_jit()
    if not no_me and name in _ACT_FN_ME:
        # If not exporting or scripting the model, first look for a memory optimized version
        # activation with custom autograd, then fallback to jit scripted, then a Python or Torch builtin
        return _ACT_FN_ME[name]
    no_jit = config.is_exportable() or config.is_no_jit()
    # NOTE: export tracing should work with jit scripted components, but I keep running into issues
    if no_jit and name in _ACT_FN_JIT:  # jit scripted models should be okay for export/scripting
        return _ACT_FN_JIT[name]
    return _ACT_FN_DEFAULT[name]


def get_act_layer(name='relu'):
    """ Activation Layer Factory
    Fetching activation layers by name with this function allows export or torch script friendly
    functions to be returned dynamically based on current config.
    """
    if name in _OVERRIDE_LAYER:
        return _OVERRIDE_LAYER[name]
    no_me = config.is_exportable() or config.is_scriptable() or config.is_no_jit()
    if not no_me and name in _ACT_LAYER_ME:
        return _ACT_LAYER_ME[name]
    no_jit = config.is_exportable() or config.is_no_jit()
    if not no_jit and name in _ACT_LAYER_JIT:  # jit scripted models should be okay for export/scripting
        return _ACT_LAYER_JIT[name]
    return _ACT_LAYER_DEFAULT[name]
