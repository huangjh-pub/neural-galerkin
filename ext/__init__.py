from pathlib import Path
import torch
from torch.utils.cpp_extension import load


def p(rel_path):
    abs_path = Path(__file__).parent / rel_path
    return str(abs_path)


_sdfgen_module = load(name='ngs_sdfgen',
                      sources=[p('sdfgen/bind.cpp'), p('../csrc/common/kdtree_cuda.cu'),
                               p('sdfgen/sdf_from_points.cu')],
                      extra_cflags=['-O2'],
                      extra_cuda_cflags=['-O2', '-Xcompiler -fno-gnu-unique'])
sdf_from_points = _sdfgen_module.sdf_from_points
