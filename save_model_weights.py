import torch
import struct
import numpy as np


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc = torch.nn.Linear(5, 1, bias=False)
        self.fc.weight.data = torch.nn.Parameter(torch.ones(5))
        self.bias = torch.nn.Parameter(torch.ones(1))

    def forward(self, x):
        x = self.fc(x) + self.bias
        return x


def parse_hparams(outfile):
    in_channels = 5
    outfile.write(struct.pack("i", in_channels))


def parse_model(checkpoint, outfile):
    """Load encodec model checkpoint."""
    n_f32 = 0

    for name in checkpoint.keys():
        var_data = checkpoint[name]
        var_data = var_data.numpy().squeeze()

        print(f"Processing variable: {name} with shape: {var_data.shape}")

        print("  Converting to float32")
        var_data = var_data.astype(np.float32)
        ftype_cur = 0
        n_f32 += 1

        n_dims = len(var_data.shape)
        encoded_name = name.encode("utf-8")
        outfile.write(struct.pack("iii", n_dims, len(encoded_name), ftype_cur))

        for i in range(n_dims):
            outfile.write(struct.pack("i", var_data.shape[n_dims - 1 - i]))

        outfile.write(encoded_name)
        var_data.tofile(outfile)

    outfile.close()

    print()
    print(f"n_f32: {n_f32}")


if __name__ == "__main__":
    model = Model().cpu()

    print(model.state_dict())

    x = torch.tensor([0, 0, 1, 2, 3], dtype=torch.float32).cpu()
    assert model(x) == x.sum() + 1

    checkpoint = model.state_dict()

    # Step 1: insert ggml magic
    outfile = open("ggml-model.bin", "wb")
    outfile.write(struct.pack("i", 0x67676D6C))

    # Step 2: insert hyperparameters
    parse_hparams(outfile)

    # Step 3: insert weights
    parse_model(checkpoint, outfile)
