# pylint: disable=global-statement,invalid-name
from typing import Any, Optional
from typing_extensions import Literal

__all__ = [
    'is_exportable', 'is_scriptable', 'is_no_jit', 'layer_config_kwargs',
    'set_exportable', 'set_scriptable', 'set_no_jit', 'set_layer_config'
]

# Set to True if prefer to have layers with no jit optimization (includes activations)
_NO_JIT = False

# Set to True if prefer to have activation layers with no jit optimization
# NOTE not currently used as no difference between no_jit and no_activation jit as only layers obeying
# the jit flags so far are activations. This will change as more layers are updated and/or added.
_NO_ACTIVATION_JIT = False

# Set to True if exporting a model with Same padding via ONNX
_EXPORTABLE = False

# Set to True if wanting to use torch.jit.script on a model
_SCRIPTABLE = False


def is_no_jit():
    """get no_jit state"""
    return _NO_JIT


class set_no_jit:
    def __init__(self, mode: bool) -> None:
        """set no_jit and save previous state"""
        global _NO_JIT
        self.prev = _NO_JIT
        _NO_JIT = mode

    def __enter__(self) -> None:
        pass

    def __exit__(self, *args: Any) -> Literal[False]:
        global _NO_JIT
        _NO_JIT = self.prev
        return False


def is_exportable():
    """get exportable state"""
    return _EXPORTABLE


class set_exportable:
    """set exportable and save previous state"""
    def __init__(self, mode: bool) -> None:
        global _EXPORTABLE
        self.prev = _EXPORTABLE
        _EXPORTABLE = mode

    def __enter__(self) -> None:
        pass

    def __exit__(self, *args: Any) -> Literal[False]:
        global _EXPORTABLE
        _EXPORTABLE = self.prev
        return False


def is_scriptable():
    """get scriptable state"""
    return _SCRIPTABLE


class set_scriptable:
    """set scriptable and save previous state"""
    def __init__(self, mode: bool) -> None:
        global _SCRIPTABLE
        self.prev = _SCRIPTABLE
        _SCRIPTABLE = mode

    def __enter__(self) -> None:
        pass

    def __exit__(self, *args: Any) -> Literal[False]:
        global _SCRIPTABLE
        _SCRIPTABLE = self.prev
        return False


class set_layer_config:
    """ Layer config context manager that allows setting all layer config flags at once.
    If a flag arg is None, it will not change the current value.
    """
    def __init__(
            self,
            scriptable: Optional[bool] = None,
            exportable: Optional[bool] = None,
            no_jit: Optional[bool] = None,
            no_activation_jit: Optional[bool] = None):
        global _SCRIPTABLE
        global _EXPORTABLE
        global _NO_JIT
        global _NO_ACTIVATION_JIT
        self.prev = _SCRIPTABLE, _EXPORTABLE, _NO_JIT, _NO_ACTIVATION_JIT
        if scriptable is not None:
            _SCRIPTABLE = scriptable
        if exportable is not None:
            _EXPORTABLE = exportable
        if no_jit is not None:
            _NO_JIT = no_jit
        if no_activation_jit is not None:
            _NO_ACTIVATION_JIT = no_activation_jit

    def __enter__(self) -> None:
        pass

    def __exit__(self, *args: Any) -> Literal[False]:
        global _SCRIPTABLE
        global _EXPORTABLE
        global _NO_JIT
        global _NO_ACTIVATION_JIT
        _SCRIPTABLE, _EXPORTABLE, _NO_JIT, _NO_ACTIVATION_JIT = self.prev
        return False


def layer_config_kwargs(kwargs):
    """ Consume config kwargs and return contextmgr obj """
    return set_layer_config(
        scriptable=kwargs.pop('scriptable', None),
        exportable=kwargs.pop('exportable', None),
        no_jit=kwargs.pop('no_jit', None))
