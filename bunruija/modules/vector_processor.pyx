# distutils: language = c++
# distutils: sources = bunruija/modules/bunruija_vector.cc
from libcpp cimport string


cdef extern from "bunruija_vector.h" namespace "bunruija":
    cdef cppclass CppPretrainedVectorProcessor "bunruija::PretrainedVectorProcessor":
        CppPretrainedVectorProcessor()
        int convert(char *)


cdef class PretrainedVectorProcessor:
    cdef CppPretrainedVectorProcessor *thisptr

    def __cinit__(self):
        self.thisptr = new CppPretrainedVectorProcessor()

    def convert(self, input_file):
        self.thisptr.convert(input_file.encode('utf-8'))
