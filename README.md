# XCube: Large-Scale 3D Generative Modeling using Sparse Voxel Hierarchies
![XCube](assets/teaser.png)

**XCube: Large-Scale 3D Generative Modeling using Sparse Voxel Hierarchies**<br>
[Xuanchi Ren](https://xuanchiren.com/),
[Jiahui Huang](https://huangjh-pub.github.io/),
[Xiaohui Zeng](https://www.cs.utoronto.ca/~xiaohui/),
[Ken Museth](https://ken.museth.org/Welcome.html),
[Sanja Fidler](https://www.cs.toronto.edu/~fidler/),
[Francis Williams](https://www.fwilliams.info/) <br>
**[Paper](https://arxiv.org/pdf/2312.03806), [Project Page](https://research.nvidia.com/labs/toronto-ai/xcube/)**

Abstract: *We present XCube (abbreviated as <span>X<sup>3</sup></span>), a novel generative model for high-resolution sparse 3D voxel grids with arbitrary attributes. Our model can generate millions of voxels with a finest effective resolution of up to <span>1024<sup>3</sup></span> in a feed-forward fashion without time-consuming test-time optimization. To achieve this, we employ a hierarchical voxel latent diffusion model which generates progressively higher resolution grids in a coarse-to-fine manner using a custom framework built on the highly efficient VDB data structure. Apart from generating high-resolution objects, we demonstrate the effectiveness of XCube on large outdoor scenes at scales of 100m x 100m with a voxel size as small as 10cm. We observe clear qualitative and quantitative improvements over past approaches. In addition to unconditional generation, we show that our model can be used to solve a variety of tasks such as user-guided editing, scene completion from a single scan, and text-to-3D.*

For business inquiries, please visit our website and submit the form: [NVIDIA Research Licensing](https://www.nvidia.com/en-us/research/inquiries/)

Please stay tuned for code and model release!

## Citation

```bibtex
@inproceedings{ren2024xcube,
    title={XCube: Large-Scale 3D Generative Modeling using Sparse Voxel Hierarchies}, 
    author={Ren, Xuanchi and Huang, Jiahui and Zeng, Xiaohui and Museth, Ken and Fidler, Sanja and Williams, Francis},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year={2024}
}
```
