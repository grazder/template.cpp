import sys
import torch
import pytest

torch.manual_seed(1337)

sys.path.append("../")

try:
    from build import bindings
except ImportError:
    raise ImportError("Please build package before running tests...")

from weights_export.export_model_weights import Model

MODEL_PATH = "example.gguf"

success, cpp_model = bindings.load_model(MODEL_PATH)
if not success:
    raise ValueError("Failed to load model!")

py_model = Model().cpu()

test_cases = [
    torch.randn(5).tolist() for _ in range(10)
]


class TestBinding:
    @torch.no_grad()
    @pytest.mark.parametrize("input_data", test_cases)
    def test_py_cpp_identity(self, input_data):
        cpp_result = torch.tensor(bindings.compute(cpp_model, input_data)[0])
        py_result = py_model(
            torch.tensor(input_data, dtype=cpp_result.dtype, device=cpp_result.device)
        )

        assert torch.allclose(cpp_result, py_result)
