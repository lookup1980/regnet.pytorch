# https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
# https://pytorch.org/tutorials/beginner/saving_loading_models.html

import io
import numpy as np

from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx

from regnet import *

model_url = 'RegNetX-200M-5e5535e1.pth'
batch_size = 1    # just a random number

torch_model = regnetx_002()
torch_model.load_state_dict(torch.load(model_url))
torch_model.eval()

# print(model_url)
# print(torch_model)

batch_size = 1
x = torch.randn(batch_size, 3, 224, 224, requires_grad=True)
torch_out = torch_model(x)
print(torch_out.shape)


# Export the model
torch.onnx.export(torch_model,               # model being run
                  # model input (or a tuple for multiple inputs)
                  x,
                  # where to save the model (can be a file or file-like object)
                  "RegNetX-200M-5e5535e1.onnx",
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['input'],   # the model's input names
                  output_names=['output'],  # the model's output names
                  dynamic_axes={'input': {0: 'batch_size'},    # variable length axes
                                'output': {0: 'batch_size'}})
