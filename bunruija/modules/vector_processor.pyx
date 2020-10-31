# distutils: language = c++
# distutils: sources = bunruija/modules/bunruija_vector.cc
from libcpp.string cimport string


cdef extern from "bunruija_vector.h" namespace "bunruija":
    cdef cppclass CppStatus "bunruija::Status":
        CppStatus() except +
        CppStatus(int, string) except +
        int status_code
        string status_message

    cdef cppclass CppPretrainedVectorProcessor "bunruija::PretrainedVectorProcessor":
        CppPretrainedVectorProcessor() except +
        CppStatus convert(char *)


cdef class Status:
    cdef CppStatus *thisptr

    def __cinit__(self, code=0, message=""):
        cdef string message_ = string(bytes(message))
        self.thisptr = new CppStatus(code, message_)

    def __repr__(self):
        msg = self.thisptr.status_message.decode('utf-8')
        return f'Status(status_code={self.thisptr.status_code}, status_message=\"{msg}\")'


cdef class PretrainedVectorProcessor:
    cdef CppPretrainedVectorProcessor *thisptr

    def __cinit__(self):
        self.thisptr = new CppPretrainedVectorProcessor()

    def convert(self, input_file):
        status_ = self.thisptr.convert(input_file.encode('utf-8'))
        status = Status(status_.status_code, status_.status_message)
        return status
