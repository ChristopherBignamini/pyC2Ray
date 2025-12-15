#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_SSIZE_T_CLEAN

#include "tests.cuh"
#include "utils.cuh"

#include <Python.h>
#include <numpy/arrayobject.h>

PyObject *asora_test_cinterp([[maybe_unused]] PyObject *self, PyObject *args) {
    PyArrayObject *dens;

    // Error checking
    if (!PyArg_ParseTuple(args, "O", &dens)) return nullptr;

    if (!PyArray_Check(dens) || PyArray_TYPE(dens) != NPY_DOUBLE ||
        PyArray_NDIM(dens) != 3) {
        PyErr_SetString(PyExc_TypeError, "dens must be numpy array of type double");
        return nullptr;
    }

    auto dens_data = static_cast<double *>(PyArray_DATA(dens));
    auto shape = PyArray_SHAPE(dens);

    auto coldens =
        reinterpret_cast<PyArrayObject *>(PyArray_SimpleNew(3, shape, NPY_DOUBLE));
    auto coldens_data = static_cast<double *>(PyArray_DATA(coldens));

    auto path =
        reinterpret_cast<PyArrayObject *>(PyArray_SimpleNew(3, shape, NPY_DOUBLE));
    auto path_data = static_cast<double *>(PyArray_DATA(path));

    // Run test kernel
    try {
        std::array<size_t, 3> cpp_shape;
        std::copy(shape, shape + 3, cpp_shape.begin());
        asoratest::cinterp_gpu(coldens_data, path_data, dens_data, cpp_shape);
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_MemoryError, e.what());
        return nullptr;
    }

    return Py_BuildValue(
        "OO",
        PyArray_Return(reinterpret_cast<PyArrayObject *>(coldens)),  //
        PyArray_Return(reinterpret_cast<PyArrayObject *>(path))      //
    );
}

PyObject *asora_test_linthrd2cart([[maybe_unused]] PyObject *self, PyObject *args) {
    int q, s;
    if (!PyArg_ParseTuple(args, "ii", &q, &s)) return nullptr;

    try {
        auto [i, j, k] = asoratest::linthrd2cart(q, s);
        return Py_BuildValue("iii", i, j, k);
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_MemoryError, e.what());
        return nullptr;
    }
}

PyObject *asora_test_cart2linthrd([[maybe_unused]] PyObject *self, PyObject *args) {
    int i, j, k;
    if (!PyArg_ParseTuple(args, "iii", &i, &j, &k)) return nullptr;

    try {
        auto [q, s] = asoratest::cart2linthrd(i, j, k);
        return Py_BuildValue("ii", q, s);
    } catch (const std::exception &e) {
        PyErr_SetString(PyExc_MemoryError, e.what());
        return nullptr;
    }
}

PyObject *asora_test_cells_in_shell([[maybe_unused]] PyObject *self, PyObject *args) {
    int q;
    if (!PyArg_ParseTuple(args, "i", &q)) return nullptr;

    auto n = asora::cells_in_shell(q);
    return Py_BuildValue("i", n);
}

PyObject *asora_test_cells_to_shell([[maybe_unused]] PyObject *self, PyObject *args) {
    int q;
    if (!PyArg_ParseTuple(args, "i", &q)) return nullptr;

    auto n = asora::cells_to_shell(q);
    return Py_BuildValue("i", n);
}

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// ========================================================================
// Define module functions and initialization function
// ========================================================================
static PyMethodDef asoraMethods[] = {
    {"cinterp", asora_test_cinterp, METH_VARARGS, "Geometric OCTA raytracing (GPU)"},
    {"linthrd2cart", asora_test_linthrd2cart, METH_VARARGS,
     "Shell indexing to cartesian coordinates"},
    {"cart2linthrd", asora_test_cart2linthrd, METH_VARARGS,
     "Cartesian coordinates to shell indexing"},
    {"cells_in_shell", asora_test_cells_in_shell, METH_VARARGS,
     "Number of cells in q-shell"},
    {"cells_to_shell", asora_test_cells_to_shell, METH_VARARGS,
     "Cumulative number of cells up to q-shell"},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

static struct PyModuleDef asoramodule = {
    PyModuleDef_HEAD_INIT, "libasoratest",
    "Exposure of internal functions for testing purposes", -1, asoraMethods
};

PyMODINIT_FUNC PyInit_libasoratest(void) {
    PyObject *module = PyModule_Create(&asoramodule);
    import_array();
    return module;
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
