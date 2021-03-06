import argparse
import json
import os
import shutil

import numpy as np
import tvm._ffi

from micro_eval import dataset
from micro_eval import model
from micro_eval import util


def generate_project(model_inst, work_tree):
    compiled = model_inst.build_model()
    lowered = model_inst.lower_model(compiled)

    standalone_crt_dir = os.path.join(os.path.dirname(tvm._ffi.libinfo.find_lib_path()[0]), "standalone_crt")
    subdirs = ["include", "src"]
    crt_root = os.path.join(work_tree, "crt")
    print('gen', os.path.join(crt_root, "model.c"))
    #TODO: fix this error
    lowered.lib.save(os.path.join(work_tree, "model.c"), fmt="cc")
    for subdir in subdirs:
      dest_dir = os.path.join(crt_root, subdir)
      if os.path.exists(dest_dir):
        shutil.rmtree(os.path.join(crt_root, subdir))

      shutil.copytree(os.path.join(standalone_crt_dir, subdir), dest_dir)

    dataset_generator_name = model_inst.dataset_generator_name()
    dataset_gen = dataset.DatasetGenerator.instantiate(dataset_generator_name, {'shuffle': False})

    samples = dataset_gen.generate(1)
    inputs = model_inst.adapt_sample_inputs(samples[0].inputs)
    with open(os.path.join(work_tree, 'inputs.c.inc'), 'w') as input_f:
      input_f.write('#include <tvm/runtime/c_runtime_api.h>\n')
      for i in inputs:
        input_num_elements = np.prod(inputs[i].data.shape)
        input_f.write(f'static const {inputs[i].data.dtype}_t input_{i}_data[{input_num_elements}] = {{\n')
        for d in inputs[i].data.flatten():
          input_f.write(f'{hex(d)},\n')
        input_f.write('};\n')

        input_f.write('\n'.join([
          f'static const int64_t input_{i}_shape[{len(inputs[i].data.shape)}] = ',
          f'     {{{", ".join([str(r) for r in inputs[i].data.shape])}}};' + '\n']))
        input_f.write('\n'.join([
          f'static const DLTensor input_{i}_tensor = {{',
          f'    (void*) input_{i}_data,',
          '    {kDLCPU, 0},',
          f'    {len(inputs[i].data.shape)},',
          '    {kDLInt, 8, 0},',
          f'    (void*) input_{i}_shape,',
          '    NULL,'
          '    0};\n']))

    with open(os.path.join(work_tree, 'graph_json.c.inc'), 'w') as json_f:
      # Remove spaces and escape quotes.
      sanitized = json.dumps(json.loads(lowered.graph_json), separators=(',', ':')).replace('"', '\\"')
      json_f.write(f'static const char* graph_json = "{sanitized}";' + '\n')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-dir', default=f'{util.get_repo_root()}/standalone', help='Path to the project to modify.')
    parser.add_argument('model_specs', nargs=1,
                        default=['cifar10_cnn:micro_dev:data/cifar10-config-validate.json'],
                        help=('Specifies the models to evaluate in terms of model name, setting, '
                              'and config. Entries are of the form '
                              '<model_name>:[setting=]<setting>[:[config=]<config>]. <model_name> '
                              'is a string naming the Python module relative to micro_eval.models '
                              'that defines the TunableModule subclass to use. <setting> describes '
                              'the target and runtime used, and is one of '
                              f'{{{",".join(model.SETTING_TO_TARGET_AND_CONTEXT)}}}'
                              '. <config> is the path to a JSON file containing tweaks to the '
                              'built module.'))

    return parser.parse_args()


def main():
    args = parse_args()
    model_inst, _ = model.instantiate_from_spec(args.model_specs[0])
    generate_project(model_inst, args.project_dir)

if __name__ == '__main__':
  main()
