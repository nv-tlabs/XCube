# Neural Kernel Surface Reconstruction

[![PyPI version](https://badge.fury.io/py/nksr.svg)](https://badge.fury.io/py/nksr)

![NKSR](assets/teaser.png)

**[Paper](https://huangjh-pub.github.io/publication/nksr/paper.pdf), [Project Page](https://research.nvidia.com/labs/toronto-ai/NKSR/)**

Abstract: *We present a novel method for reconstructing a 3D implicit surface from a large-scale, sparse, and noisy point cloud. 
Our approach builds upon the recently introduced [Neural Kernel Fields (NKF)](https://nv-tlabs.github.io/nkf/) representation. 
It enjoys similar generalization capabilities to NKF, while simultaneously addressing its main limitations: 
(a) We can scale to large scenes through compactly supported kernel functions, which enable the use of memory-efficient sparse linear solvers. 
(b) We are robust to noise, through a gradient fitting solve. 
(c) We minimize training requirements, enabling us to learn from any dataset of dense oriented points, and even mix training data consisting of objects and scenes at different scales. 
Our method is capable of reconstructing millions of points in a few seconds, and handling very large scenes in an out-of-core fashion. 
We achieve state-of-the-art results on reconstruction benchmarks consisting of single objects, indoor scenes, and outdoor scenes.*

For business inquiries, please visit our website and submit the
form: [NVIDIA Research Licensing](https://www.nvidia.com/en-us/research/inquiries/)

## Usage

Please refer to the [main repository](https://github.com/nv-tlabs/nksr) for details.

## Citation

```bibtex
@inproceedings{huang2023nksr,
  title={Neural Kernel Surface Reconstruction},
  author={Huang, Jiahui and Gojcic, Zan and Atzmon, Matan and Litany, Or and Fidler, Sanja and Williams, Francis},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4369--4379},
  year={2023}
}
```
