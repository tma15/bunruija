# distutils: language = c++
from libcpp cimport bool
from libcpp.string cimport string


cdef extern from "bunruija/string_projector_op.h" namespace "bunruija":
    cdef cppclass CppStringProjectorOp "bunruija::StringProjectorOp":
        CppStringProjectorOp(int feature_size) except +
        bool is_training()
        int call "operator()" (string)


cdef class StringProjectorOp:
    cdef CppStringProjectorOp *thisptr

    def __init__(self, feature_size):
        self.thisptr = new CppStringProjectorOp(feature_size)

    def __call__(self, word):
        return self.thisptr.call(word.encode('utf8'))
