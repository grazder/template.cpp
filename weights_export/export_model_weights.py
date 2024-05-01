import torch
import struct
import numpy as np
from gguf import GGUFWriter

torch.manual_seed(52)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc = torch.nn.Linear(5, 1, bias=False)
        self.fc.weight.data = torch.nn.Parameter(torch.ones(5))
        self.bias = torch.nn.Parameter(torch.ones(1))

    def forward(self, x):
        x = self.fc(x) + self.bias
        return x


def parse_hparams(gguf_writer):
    in_channels = 5
    bias_size = 1
    gguf_writer.add_int32("in_channels", in_channels)
    gguf_writer.add_int32("bias_size", bias_size)


def parse_model(checkpoint, gguf_writer):
    """Load encodec model checkpoint."""
    for name in checkpoint.keys():
        var_data = checkpoint[name]
        var_data = var_data.numpy().squeeze().astype(np.float32)
        gguf_writer.add_tensor(name, var_data)

        print(f"Processing variable: {name} with shape: {var_data.shape}")


if __name__ == "__main__":
    model = Model().cpu()

    print(model.state_dict())

    x = torch.tensor([0, 0, 1, 2, 3], dtype=torch.float32).cpu()
    assert model(x) == x.sum() + 1

    checkpoint = model.state_dict()

    gguf_writer = GGUFWriter("example.gguf", "linear")

    # Step 2: insert hyperparameters
    parse_hparams(gguf_writer)

    # Step 3: insert weights
    parse_model(checkpoint, gguf_writer)

    # Step 4: saving model and hparams to file
    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()

    gguf_writer.close()
