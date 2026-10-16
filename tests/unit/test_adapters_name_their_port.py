"""Every adapter class Otari ships names the port or session protocol it satisfies.

A named base makes mypy name a missing method in its error.
"""

import importlib
import inspect
import pkgutil
from typing import is_protocol

import pytest

import gateway.adapters
import gateway.ports
from gateway.adapters.file_storage_adapter import LocalDirFileStore


def _adapter_classes() -> list[type]:
    classes: list[type] = []
    for module_info in pkgutil.iter_modules(gateway.adapters.__path__):
        module = importlib.import_module(f"{gateway.adapters.__name__}.{module_info.name}")
        classes.extend(
            cls for _, cls in inspect.getmembers(module, inspect.isclass) if cls.__module__ == module.__name__
        )
    return sorted(classes, key=lambda cls: cls.__qualname__)


def _is_port(cls: type) -> bool:
    return cls.__module__.startswith(f"{gateway.ports.__name__}.") and is_protocol(cls)


def _names_a_port(cls: type) -> bool:
    return any(_is_port(base) for base in cls.__bases__)


@pytest.mark.parametrize("adapter", _adapter_classes(), ids=lambda cls: cls.__qualname__)
def test_adapter_names_its_port(adapter: type) -> None:
    assert _names_a_port(adapter), f"{adapter.__qualname__} names no port as a base"


def test_a_class_that_inherits_its_port_through_an_adapter_does_not_name_it() -> None:
    class Indirect(LocalDirFileStore):
        pass

    assert not _names_a_port(Indirect)
