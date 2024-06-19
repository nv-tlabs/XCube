import re
import os
import subprocess
import sys
from setuptools import find_packages, setup
from torch.utils.cpp_extension import (CUDA_HOME, BuildExtension, CUDAExtension)
from fvdb.utils.build_ext import FVDBExtension

with open("nksr/__init__.py", "r") as fh:
    __version__ = re.findall(r'__version__ = \'(.*?)\'', fh.read())[0]


if CUDA_HOME is None:
    print("Please install nvcc for your PyTorch distribution and set CUDA_HOME environment variable.")
    sys.exit(-1)


if sys.platform != "linux":
    print("This repository only supports x86-64 Linux!")


def download_external_dep(name: str, git_url: str, git_tag: str,
                          recursive: bool = False, setuptools_install: bool = False, checkout: bool = True):
    import git
    from git.exc import InvalidGitRepositoryError

    based = os.path.dirname(os.path.abspath(__file__))
    external_path = os.path.join(based, 'external')
    if not os.path.exists(external_path):
        os.makedirs(external_path, exist_ok=True)
    elif not os.path.isdir(external_path):
        raise RuntimeError(f"External path {external_path} exists but is not a directory")

    repo_path = os.path.join(external_path, name)
    if os.path.exists(repo_path) and os.path.isdir(repo_path):
        if checkout:
            try:
                repo = git.Repo(repo_path)
                repo.git.checkout(git_tag)
            except InvalidGitRepositoryError:
                raise ValueError(f"A path {repo_path} exists but is not a git repo")
    else:
        if recursive:
            repo = git.Repo.clone_from(git_url, repo_path, multi_options=['--recursive'])
        else:
            repo = git.Repo.clone_from(git_url, repo_path)
        repo.git.checkout(git_tag)

    if setuptools_install:
        res = subprocess.Popen(['python', 'setup.py', sys.argv[1]], cwd=repo_path)
        res.communicate()
        assert res.returncode == 0, "submodule install failure!"


download_external_dep(
    name='eigen',
    git_url='https://gitlab.com/libeigen/eigen.git',
    git_tag='3.4'
)

cwd = os.path.dirname(os.path.abspath(__file__))
extra_includes = [
    os.path.join(cwd, "external/eigen")
]

def get_source_files(base_path):
    all_sources = [os.path.join(base_path, fname) for fname in os.listdir(base_path)]
    return [t for t in all_sources if t.endswith(".cu") or t.endswith(".cpp")]


bind_ext = FVDBExtension(
    name='nksr._C',
    sources=['csrc/bind.cpp'] +
            get_source_files("csrc/kernel_eval") +
            get_source_files("csrc/meshing") +
            get_source_files("csrc/pcproc") +
            get_source_files("csrc/sparse_solve"),
    include_dirs=extra_includes,
    extra_compile_args={'cxx': ['-O2'], 'nvcc': ['-O2']},
    libraries=['cusparse']
)


setup(
    name='nksr',
    version=__version__,
    description='Neural Kernel Surface Reconstruction',
    author_email='huangjh.work@outlook.com',
    keywords=['nksr', '3d', 'reconstruction'],
    python_requires='>=3.7',
    install_requires=['pykdtree', 'torch', 'python-pycg'],
    ext_modules=[bind_ext],
    cmdclass={
        'build_ext':
        BuildExtension.with_options(no_python_abi_suffix=True, use_ninja=True)
    },
    packages=find_packages(),
    include_package_data=True,
)
