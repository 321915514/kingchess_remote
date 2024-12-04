import os
import shutil
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from pathlib import Path

import onnx
# from policy_value_net_pytorch import PolicyValueNet
import torch.onnx
from policy_value_net_pytorch import Net
import argparse


def get_latest_opset():
    """Return second-most (for maturity) recently supported ONNX opset by this version of torch."""
    a = max(int(k[14:]) for k in vars(torch.onnx) if "symbolic_opset" in k) - 1
    # print(a)
    return max(int(k[14:]) for k in vars(torch.onnx) if "symbolic_opset" in k) - 1  # opset



# if __name__ == '__main__':
#     # os.environ['CUDA_VISIBLE_DEVICES'] = '8'
#     model_path = './model/pytorch.pt'
#     # model_path = '/home/dev/board_seg.pth'
#     net = Net()

def export_engine(model_path, onnx_file, trt_file):
    f_onnx, _ = export_onnx(onnx_file, model_path)
    try:
        import tensorrt as trt
    except ImportError:
        print('没有 tensorrt 库')
        return
    assert Path(f_onnx).exists(), 'f failed to export onnx file'
    logger = trt.Logger(trt.Logger.INFO)


    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network,logger)
    if not parser.parse_from_file(f_onnx):
        raise RuntimeError('failed load ONNX file')

    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]

    for inp in inputs:
        print(f'engine input "{inp.name}" with shape{inp.shape} {inp.dtype}')
    for out in outputs:
        print(f'engine output "{out.name}" with shape{out.shape} {out.dtype}')

    im = torch.zeros(1, 32, 5, 9).to('cuda')
    # im = torch.zeros((1, 405)).to('cuda:0')

    shape = im.shape

    profile = builder.create_optimization_profile()
    for inp in inputs:
        profile.set_shape(inp.name, (1, *shape[1:]), (max(1, shape[0]//2), *shape[1:]), shape)
    config.add_optimization_profile(profile)

    print(
        f"engine building FP{32} engine as {trt_file}"
    )
    torch.cuda.empty_cache()

    # Write file
    with builder.build_engine(network, config) as engine, open(trt_file, "wb") as t:
        t.write(engine.serialize())
    # 修改后
    #with builder.build_serialized_network(network, config) as engine, open(trt_file, "wb") as t:
    #    t.write(engine.serialize())

def export_onnx(f: str, model_path):
    net = Net()
    if not os.path.exists(model_path):
        torch.save(net.state_dict(),model_path)
        net.load_state_dict(torch.load(model_path))
        print('重新初始化模型！！！')
    else:
        net.load_state_dict(torch.load(model_path, map_location='cpu'))
    im = torch.zeros(1, 32, 5, 9)
    # linear
    # im = torch.zeros((1, 405))
    torch.onnx.export(
        net,  # dynamic=True only compatible with cpu
        im,
        f,
        verbose=False,
        opset_version=14,
        do_constant_folding=True,  # WARNING: DNN inference with torch>=1.12 may require do_constant_folding=False
        input_names=["board"],
        output_names=["policy", "value"],
        dynamic_axes={"board": {0: "batch_size"}}
    )
    model_onnx = onnx.load(f)
    return f, model_onnx


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # if not os.path.dirname("model"):
    #     os.makedirs('model', exist_ok=True)
    

    parse = argparse.ArgumentParser()

    parse.add_argument('--model',type=str,required=True)
    

    args = parse.parse_args()

    model_path = args.model
    #model_path = './model/26000.pth'
    #shutil.copy(model_path, './supervise_model/current.pth')

    export_engine(model_path, './model/current.onnx', './model/current.engine')

