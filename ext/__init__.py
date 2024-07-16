import glob
import os.path
from pathlib import Path
import torch
from torch.utils.cpp_extension import load


def load_torch_extension(name, additional_files=None, ignore_files=None, **kwargs):
    if ignore_files is None:
        ignore_files = []

    if additional_files is None:
        additional_files = []

    def path_should_keep(pth):
        for file_name in ignore_files:
            if file_name in pth:
                return False
        return True

    base_path = Path(__file__).parent / name
    cpp_files = glob.glob(str(base_path / "*.cpp"), recursive=True)
    cpp_files = filter(path_should_keep, cpp_files)
    cu_files = glob.glob(str(base_path / "*.cu"), recursive=True)
    cu_files = filter(path_should_keep, cu_files)

    # Sanitize the name to avoid special characters
    sanitized_name = name.replace(".", "_").replace("+", "_")

    # Constructing the extension name, removing special characters
    extension_name = f"xcube_torch_{torch.__version__}".replace(".", "_").replace("+", "_") + "_" + sanitized_name

    return load(
        name=extension_name,
        sources=list(cpp_files) + list(cu_files) + [base_path / t for t in additional_files],
        verbose='COMPILE_VERBOSE' in os.environ.keys(),
        **kwargs
    )

common = load_torch_extension(
    'common', extra_cflags=['-O2'], extra_cuda_cflags=['-O2', '-Xcompiler -fno-gnu-unique']
)

sdfgen = load_torch_extension(
    'sdfgen',
    ignore_files=['sdf_from_mesh', 'triangle_bvh'],
    additional_files=['../common/kdtree_cuda.cu'],
    extra_cflags=['-O2'],
    extra_cuda_cflags=['-O2', '-Xcompiler -fno-gnu-unique'],
)