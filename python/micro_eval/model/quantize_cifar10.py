import argparse
import os

from mxnet import nd, autograd, gluon
from mxnet.gluon.data.vision import transforms
import onnx
import tvm
from tvm import relay
import numpy as np

from micro_eval import util
from micro_eval.util import model_util


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx-model', default=f'/data/cifar10.onnx',
                        help='path to unquantized cifar10 model')
    parser.add_argument('--num-samples', default=1024, type=int,
                        help='number of samples to use for data-aware quantization')
    parser.add_argument('--quantized-tvm-model',
                        default=f'/data/quantized-cifar10.onnx',
                        help='path to write quantized cifar10 model')
    parser.add_argument('--test-accuracy', action='store_true',
                        help='test accuracy of the quantized model')

    args = parser.parse_args()

    onnx_model = onnx.load(args.onnx_model)

    # get module from original model
    data_shape = util.LabelledShape(N=1, C=3, H=32, W=32, dtype='uint8')
    mod, params = relay.frontend.from_onnx(onnx_model, {"data": data_shape.shape}, freeze_params=True)

    # get samples for quantization
    samples = model_util.get_sample_points(args.num_samples, data_shape.layout)
    numpy_samples = []
    for data in samples:
        numpy_samples.append({'data': data['data'].data})
    # print(numpy_samples)

    # quantize
    with relay.quantize.qconfig(calibrate_mode='global_scale', global_scale=8.0,
        skip_conv_layers=[], skip_dense_layer=False,
        weight_scale='power2', partition_conversions='disabled',
        nbit_input=8, dtype_input="int8", nbit_weight=8, dtype_weight="int8",
        nbit_activation=8, dtype_activation="int8"):
        mod_quantized = relay.quantize.quantize(mod, params, dataset=numpy_samples)

    with open(args.quantized_tvm_model, 'w') as f:
    #   f.write(tvm.ir.save_json(mod_quantized))
        f.write(str(mod_quantized))

    # test accuracy
    if args.test_accuracy:
        target = "llvm"
        with tvm.transform.PassContext(opt_level=1):
            intrp = relay.build_module.create_executor("graph", mod, tvm.cpu(0), target)

        with tvm.transform.PassContext(opt_level=1):
            intrp_quantized = relay.build_module.create_executor("graph", mod_quantized, tvm.cpu(0), target)
        
        num_correct = 0
        counter = 0
        for ind, data in enumerate(samples):
            print(ind)
            val = data['data'].data
            output = intrp.evaluate()(tvm.nd.array(val.astype("float32")), **params).asnumpy()
            quantized_output = intrp.evaluate()(tvm.nd.array(val.astype("float32")), **params).asnumpy()
            if np.argmax(output) == np.argmax(quantized_output):
                num_correct += 1
            counter += 1
        print("accuracy: {}%".format(100*(num_correct/args.num_samples)))

    # build
    build_dir = 'build'
    if not os.path.isdir(build_dir):
        os.mkdir(build_dir)

    # with tvm.transform.PassContext(opt_level=3):
    #     garph, lib, params = relay.build_module.build(mod, 'llvm', params=params, mod_name='cifar10')
    
    # lib.save(os.path.join(build_dir, 'model.o'))
    # with open(os.path.join(build_dir, 'graph.json'), 'w') as f:
    #     f.write(garph)
    # with open(os.path.join(build_dir, 'params.bin'), 'wb') as f:
    #     f.write(relay.save_param_dict(params))

    with tvm.transform.PassContext(opt_level=3):
        garph_quantized, lib_quantized, params_quantized = relay.build_module.build(mod_quantized, 
        'llvm', params=params, mod_name='cifar10-quantized')

    lib_quantized.save(os.path.join(build_dir, 'quantized_model.o'))
    with open(os.path.join(build_dir, 'graph_quantized.json'), 'w') as f:
        f.write(garph_quantized)
    with open(os.path.join(build_dir,'params_quantized.bin'), 'wb') as f:
        f.write(relay.save_param_dict(params_quantized))

if __name__ == '__main__':
  main()
