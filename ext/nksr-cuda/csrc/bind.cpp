#include <torch/extension.h>

void pybind_kernel_eval(py::module& m);
void pybind_meshing(py::module& m);
void pybind_pcproc(py::module& m);
void pybind_sparse_solve(py::module& m);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::module m_kernel_eval = m.def_submodule("kernel_eval");
    pybind_kernel_eval(m_kernel_eval);

    py::module m_meshing = m.def_submodule("meshing");
    pybind_meshing(m_meshing);

    py::module m_pcproc = m.def_submodule("pcproc");
    pybind_pcproc(m_pcproc);

    py::module m_sparse_solve = m.def_submodule("sparse_solve");
    pybind_sparse_solve(m_sparse_solve);
}
