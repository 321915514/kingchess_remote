import os
import time
from collections import namedtuple, OrderedDict

import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
cuda.init()
# cfx = cuda.Device(0).make_context()

import pycuda.autoinit

import torch
from typing import Union, Optional, Sequence, Dict, Any
import torchvision.transforms as transforms
from PIL import Image


def load_engine(engine_file_path):
    device = torch.device('cuda:0')
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)
    assert os.path.exists(engine_file_path)
    print("Reading engine from file {}".format(engine_file_path))
    f = open(engine_file_path, "rb")
    engine = runtime.deserialize_cuda_engine(f.read())
    # get engine
    # 创建 TensorRT 的执行上下文
    # 1.创建一个Binding对象，该对象包含'name', 'dtype', 'shape', 'data', 'ptr'这些属性
    Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
    context = engine.create_execution_context()
    bindings = OrderedDict()
    output_names = []
    dynamic = False
    fp16 = False
    for i in range(engine.num_bindings):
        name = engine.get_binding_name(i)  # 获得输入输出的名字"images","output0"
        dtype = trt.nptype(engine.get_binding_dtype(i))
        if engine.binding_is_input(i):  # 判断是否为输入
            if -1 in tuple(engine.get_binding_shape(i)):  # dynamic get_binding_shape(0)->(1,3,640,640) get_binding_shape(1)->(1,25200,85)
                dynamic = True
                context.set_binding_shape(i, tuple(engine.get_profile_shape(0, i)[2]))
            if dtype == np.float16:
                fp16 = True
        else:  # output
            output_names.append(name)
        shape = tuple(context.get_binding_shape(i))  # 记录输入输出shape
        im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)  # 创建一个全0的与输入或输出shape相同的tensor
        bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))  # 放入之前创建的对象中
    binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())  # 提取name以及对应的Binding
    # batch_size = bindings['board'].shape[0]  # if dynamic, this is instead max batch size

    image = torch.from_numpy(np.zeros((1, 5, 9, 9), dtype=trt.nptype(engine.get_binding_dtype(0)))).to(device)

    s = bindings['board'].shape # 10x5x9x9
    assert image.shape == s, f"input size {image.shape} {'>' if dynamic else 'not equal to'} max model size {s}"
    binding_addrs['board'] = int(image.data_ptr())
    # 调用计算核心执行计算过程
    # cfx.push()
    # context.execute_async_v2(list(binding_addrs.values()))
    context.execute_v2(list(binding_addrs.values()))
    # cfx.pop()
    y = [bindings[x].data for x in sorted(output_names)]

    if isinstance(y, (list, tuple)):
        if len(y) == 1:
            result = y[0].to('cpu').numpy()
        else:
            result = []
            for x in y:
                result.append(x.to('cpu').numpy())

        print(result)






if __name__ == '__main__':
    # load_engine('./pytorch_para.trt')
    pass
    # load_engine("./pytorch_8.5.3.trt")

    # h_output = inference()

    # print(h_output)
