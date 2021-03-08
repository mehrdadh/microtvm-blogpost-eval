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
from . import CompiledModel, LoweredModule, TunableModel


def get_cifar10_samples(num_samples):
    data_shape = util.LabelledShape(N=1, C=3, H=32, W=32, dtype='uint8')
    samples = model_util.get_sample_points(num_samples, data_shape.layout)
    numpy_samples = []
    for data in samples:
        numpy_samples.append({'data': data['data'].data})
    return numpy_samples

class Cifar10ONNX(TunableModel):
    
    def _lower_micro_dev(self, compiled_model):
        with tvm.transform.PassContext(opt_level=3, config={'tir.disable_vectorize': True}):
            return tvm.relay.build(
                compiled_model.ir_mod[compiled_model.entry_point], target=self.target,
                params=compiled_model.params)

    def _lower_cpu(self, compiled_model):
        with tvm.transform.PassContext(opt_level=3):
            return tvm.relay.build(compiled_model.ir_mod[compiled_model.entry_point],
                                   target="llvm", params=compiled_model.params)

    def _quantize(self, num_samples, mod, params):
        cifar_samples = get_cifar10_samples(num_samples)
        with relay.quantize.qconfig(calibrate_mode='global_scale', global_scale=8.0,
                                    skip_conv_layers=[], skip_dense_layer=False,
                                    weight_scale='power2', partition_conversions='disabled',
                                    nbit_input=8, dtype_input="int8", nbit_weight=8, dtype_weight="int8",
                                    nbit_activation=8, dtype_activation="int8"):
            relay_mod = relay.quantize.quantize(mod, params, dataset=cifar_samples)
        return relay_mod

    def dataset_generator_name(self):
        return 'cifar10'

    def get_config_str(self):
        raise NotImplementedError()

    def lower_model(self, compiled_model):
        # Normally we would examine target to determine how to lower, but target does not currently
        # adequately describe the runtime environment.
        if self.ctx_str == 'cpu':
            return self._lower_cpu(compiled_model)
        elif self.ctx_str == 'micro_dev':
            return self._lower_micro_dev(compiled_model)
        else:
            assert False, f"don't know how to lower for context {self.ctx_str}"

    def get_micro_compiler_opts(self):
        opts = tvm.micro.default_options(os.path.join(util.get_zephyr_project_root(), 'crt'))
        opts['lib_opts']['include_dirs'] += INCLUDE_PATHS
#        opts['bin_opts']['include_dirs'] += INCLUDE_PATHS
        opts['generated_lib_opts'] = copy.copy(tvm.micro.build._CRT_GENERATED_LIB_OPTIONS)
        opts['generated_lib_opts']['include_dirs'] += INCLUDE_PATHS
        opts['generated_lib_opts']['cflags'] += ['-Wno-error=strict-aliasing']
        return opts

    def extract_tunable_tasks(self, compiled_model):
        raise NotImplementedError()

        with tvm.transform.PassContext(opt_level=3):
            tasks = tvm.autotvm.task.extract_from_program(
                compiled_model.ir_mod[compiled_model.entry_point],
                compiled_model.params,
                self.target)

        assert len(tasks) == 3
        return tasks

    def get_autotvm_measure_option(self, num_runners : int, tracker_host : str, tracker_port : int,
                                   tracker_key : str, dev_config : dict, task_index : int,
                                   task : tvm.autotvm.task.Task):
        raise NotImplementedError()

        builder = tvm.autotvm.LocalBuilder(
            build_func=tvm.micro.cross_compiler(
                dev_config,
                tvm.micro.LibType.OPERATOR,
                lib_headers=HEADERS,
                lib_include_paths=INCLUDE_PATHS),
            n_parallel=num_runners)
        builder.build_kwargs.setdefault('build_option', {})['disable_vectorize'] = True
        runner = tvm.autotvm.RPCRunner(
            tracker_key, tracker_host, tracker_port, n_parallel=num_runners,
            number=1, repeat=1, timeout=0)

        return tvm.autotvm.measure_option(builder=builder, runner=runner)

    def build_model(self):
        # model_path = '/Users/mehrdadh/work/microtvm-blogpost-eval/data/cifar10.onnx'
        # num_samples = 10

        onnx_path = self.config['onnx_path']
        onnx_model = onnx.load(f'{util.get_repo_root()}/{onnx_path}')
        data_shape = util.LabelledShape(N=1, C=3, H=32, W=32, dtype='uint8')
        mod, params = relay.frontend.from_onnx(onnx_model, {"data": data_shape.shape}, freeze_params=True)
        
        relay_mod = mod

        # quantize
        if self.config['quantize']:
            relay_mod = self._quantize(self.config['quantization_num_samples'], mod, params)

        return CompiledModel(self.target, relay_mod, params, 'main', config={})

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
