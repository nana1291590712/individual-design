# model_test_shape.py
import torch
from model import MultiScale1DCNN


if __name__ == "__main__":

    model = MultiScale1DCNN()
    model.eval()

    dummy_input = torch.randn(8, 1, 1024)

    with torch.no_grad():
        fault_out, severity_out = model(dummy_input)

    print("Input shape      :", dummy_input.shape)
    print("Fault output     :", fault_out.shape)
    print("Severity output  :", severity_out.shape)
    print("Total parameters :", sum(p.numel() for p in model.parameters()))
