# cython: c_string_type=unicode, c_string_encoding=utf8
# distutils: language = c++
import cython
from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector

cimport numpy as np
import numpy as np

np.import_array()


cdef extern from "bunruija/string_projector_op.h" namespace "bunruija":
    cdef cppclass CppStringProjectorOp "bunruija::StringProjectorOp":
        CppStringProjectorOp(int feature_size) except +
        bool is_training()
        void call "operator()" (vector[vector[string]], float *)


@cython.auto_pickle(True)
cdef class StringProjectorOp:
    cdef CppStringProjectorOp *thisptr
    cpdef int feature_size

    def __init__(self, feature_size):
        self.feature_size = feature_size
        self.thisptr = new CppStringProjectorOp(feature_size)

    def __call__(self, batch_words):
        bsz = len(batch_words)
        max_num_words = max(len(words) for words in batch_words)

        cdef np.ndarray[np.float32_t, ndim=3] projection = np.zeros(
                (bsz, self.feature_size, max_num_words), dtype=np.float32)

        self.thisptr.call(batch_words, &projection[0, 0, 0])
        return projection.reshape(bsz, max_num_words, self.feature_size)

    def __getstate__(self):
        return {}

    def __getnewargs__(self):
        return self.feature_size,

    def __setstate__(self, state):
        for k, v in state.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def __reduce_ex__(self, proto):
        func = StringProjectorOp
        args = self.__getnewargs__()
        state = self.__getstate__()
        listitems = None
        dictitems = None
        return (func, args, state, listitems, dictitems)
