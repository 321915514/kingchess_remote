import os
import time
from collections import OrderedDict, namedtuple

import numpy as np
import torch
from tensorrt import DataType

from fundamental.board import GameState
import tensorrt as trt
import pycuda.driver as cuda

from net.encoder import encoder_board

cuda.init()

import logging
logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.INFO)
logger = logging.getLogger(__file__)


class Trt:
    def __init__(self, trt_file: str, device='cuda:8'):
        self.trt_file = trt_file
        self.device = device
        self.dtypeMapping = {
            trt.int8: np.int8,
            trt.int32: np.int32,
            trt.float16: np.float16,
            trt.float32: np.float32
        }
        self.binding_addrs, self.context, self.bindings, self.output_names, self.engine, self.dynamic = self.load_engine(
            self.trt_file)
        # self.dtype = np.float32



    def load_engine(self, trt_file):
        # device = torch.device(self.device)
        # TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        # runtime = trt.Runtime(TRT_LOGGER)
        # assert os.path.exists(trt_file)
        # print("Reading engine from file {}".format(trt_file))
        # f = open(trt_file, "rb")
        # engine = runtime.deserialize_cuda_engine(f.read())
        # # get engine
        # # 创建 TensorRT 的执行上下文
        # # 1.创建一个Binding对象，该对象包含'name', 'dtype', 'shape', 'data', 'ptr'这些属性
        # Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        # context = engine.create_execution_context()
        # bindings = OrderedDict()
        # output_names = []
        # dynamic = False
        # fp16 = False
        # dtype = None

        # for i in range(engine.num_bindings):
        #     name = engine.get_binding_name(i)  # 获得输入输出的名字"images","output0"
        #     dtype = trt.nptype(engine.get_binding_dtype(i))
        #     if engine.binding_is_input(i):  # 判断是否为输入
        #         if -1 in tuple(engine.get_binding_shape(
        #                 i)):  # dynamic get_binding_shape(0)->(1,3,640,640) get_binding_shape(1)->(1,25200,85)
        #             dynamic = True
        #             context.set_binding_shape(i, tuple(engine.get_profile_shape(0, i)[2]))
        #         if dtype == np.float16:
        #             fp16 = True
        #     else:  # output
        #         output_names.append(name)
        #     shape = tuple(context.get_binding_shape(i))  # 记录输入输出shape
        #     im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)  # 创建一个全0的与输入或输出shape相同的tensor
        #     bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))  # 放入之前创建的对象中
        # binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())  # 提取name以及对应的Binding

        device = torch.device('cuda:0')
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)
        assert os.path.exists(trt_file)
        logger.info("Reading engine from file {}".format(trt_file))
        f = open(trt_file, "rb")
        engine = runtime.deserialize_cuda_engine(f.read())
        # get engine
        # 创建 TensorRT 的执行上下文
        # 1.创建一个Binding对象，该对象包含'name', 'dtype', 'shape', 'data', 'ptr'这些属性
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        context = engine.create_execution_context()
        bindings = OrderedDict()
        output_names = []
        dynamic = False
        # fp16 = False
        dtype = None
        for i in range(engine.num_bindings):
            name = engine.get_binding_name(i)  # 获得输入输出的名字"images","output0"
            # print(engine.get_binding_dtype(i))
            dtype = trt.nptype(engine.get_binding_dtype(i))
            # print(dtype)
            if engine.binding_is_input(i):  # 判断是否为输入
                if -1 in tuple(engine.get_binding_shape(
                        i)):  # dynamic get_binding_shape(0)->(1,3,640,640) get_binding_shape(1)->(1,25200,85)
                    dynamic = True
                    context.set_binding_shape(i, tuple(engine.get_profile_shape(0, i)[2]))
                # if dtype == np.float16:
                #     fp16 = True
            else:  # output
                output_names.append(name)
            shape = tuple(context.get_binding_shape(i))  # 记录输入输出shape
            im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)  # 创建一个全0的与输入或输出shape相同的tensor
            bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))  # 放入之前创建的对象中
        binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())  # 提取name以及对应的Binding

        return binding_addrs, context, bindings, output_names, engine, dynamic, dtype

    def policy_value_fn(self, game: GameState):

        # pre
        legal_positions = game.legal_moves()
        current_state = np.ascontiguousarray(encoder_board(game).reshape(-1, 5, 9, 9)).astype('float32')  # .reshape(-1, 3)
        # end
        image = torch.from_numpy(current_state).to(self.device)
        s = self.bindings['board'].shape  # 10x5x9x9
        # assert image.shape == s, f"input size {image.shape} {'>' if self.dynamic else 'not equal to'} max model
        # size {s}"
        self.binding_addrs['board'] = int(image.data_ptr())
        # 调用计算核心执行计算过程
        ## execute_async_v2
        # self.context.execute_v2(list(self.binding_addrs.values()))
        # start = time.time()
        self.context.execute_v2(list(self.binding_addrs.values()))
        # end = time.time()-start
        # logger.info(f"Inference time: {end:.6f} seconds")
        y = [self.bindings[x].data for x in sorted(self.output_names)]

        if isinstance(y, (list, tuple)):
            # if len(y) == 1:
            #     result = y[0].cpu()
            # else:
            result = []
            for x in y:
                result.append(x.cpu())

            print(result)

            log_act_probs, value = result

            act_probs = np.exp(log_act_probs.detach().numpy().astype('float32').flatten())

            act_probs = zip(legal_positions, act_probs[:len(legal_positions)])

            return act_probs, value.detach().numpy()


if __name__ == '__main__':
    trt = Trt("./pytorch_8.4.1.trt")

    game = GameState.new_game(5, 9)

    act_probs, value = trt.policy_value_fn(game)

    for move, prob in act_probs:
        print("move:{},prob:{}".format(move, prob))

    print(value)

