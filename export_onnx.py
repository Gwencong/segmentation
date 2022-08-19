import os
import sys
import onnx
import onnxruntime
import argparse
import numpy as np
import paddle

sys.path.append('/home/gwc/gwc/code/PaddleSeg')
from paddleseg.cvlibs import Config
from paddleseg.utils import logger, utils


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument(
        "--config", 
        default='configs/pp_liteseg/pp_liteseg_stdc2_custom_1024x512_scale1.0_160k.yml',
        type=str,
        help="The config file.")
    parser.add_argument(
        "--model_path", 
        default='out/train/custom_dataset/pp_litesegB/best_model/model.pdparams',
        type=str,
        help="The pretrained weights file.")
    parser.add_argument(
        '--save_dir',
        default='out/export/pp_liteseg',
        type=str,
        help='The directory for saving the predict result.')
    parser.add_argument('--width', default=1280, type=int, help='width')
    parser.add_argument('--height', default=720, type=int, help='height')
    parser.add_argument('--batch-size', default=1, type=int, help='batch size')
    parser.add_argument('--dynamic', default=False, action='store_true', help='dynamic shape')
    parser.add_argument('--print_model', default=False,action='store_true', help='print model to log')
    args = parser.parse_args()

    # 1. prepare
    cfg = Config(args.config)
    model = cfg.model
    utils.load_entire_model(model, args.model_path)
    logger.info('Loaded trained params of model successfully')

    model.eval()
    if args.print_model:
        print(model)

    input_shape = [args.batch_size, 3, args.height, args.width]
    print("input shape:", input_shape)
    input_data = np.random.random(input_shape).astype('float32')
    model_name = os.path.basename(args.config).split(".")[0]

    # 2. run paddle
    model.eval()
    paddle_out = model(paddle.to_tensor(input_data))[0].numpy()
    if paddle_out.ndim == 3:
        paddle_out = paddle_out[np.newaxis, :]
    print("out shape:", paddle_out.shape)
    print("The paddle model has been predicted by PaddlePaddle.\n")

    # 3. export onnx
    if args.dynamic:
        input_spec = paddle.static.InputSpec([args.batch_size,3,None,None], 'float32', 'x')
    else:
        input_spec = paddle.static.InputSpec(input_shape, 'float32', 'x')
    onnx_model_path = os.path.join(args.save_dir, model_name + "_model")
    paddle.onnx.export(
        model, onnx_model_path, input_spec=[input_spec], opset_version=11)
    print("Completed export onnx model.\n")

    # 4. run onnx
    onnx_model_path = onnx_model_path + ".onnx"
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)
    print('The onnx model has been checked.')

    ort_sess = onnxruntime.InferenceSession(onnx_model_path,providers=['CPUExecutionProvider'])
    ort_inputs = {ort_sess.get_inputs()[0].name: input_data}
    onnx_out = ort_sess.run(None, ort_inputs)[0]
    print("The onnx model has been predicted by ONNXRuntime.")

    # 5. check output
    assert onnx_out.shape == paddle_out.shape
    np.testing.assert_allclose(onnx_out, paddle_out, rtol=0, atol=1e-03)
    print("The paddle and onnx models have the same outputs.\n")