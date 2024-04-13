import sys

sys.path.append("../")

try:
    from build import py_bindings
except ImportError as e:
    raise ImportError("Please build package before running tests...")


print(py_bindings.add(1, 2))
