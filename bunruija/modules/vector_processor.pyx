# distutils: language = c++
from functools import lru_cache

import cython
from libcpp cimport bool
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector

import numpy as np


cdef extern from "bunruija/status.h" namespace "bunruija":
    cdef cppclass CppStatus "bunruija::Status":
        CppStatus() except +
        CppStatus(int, string) except +
        int status_code
        string status_message

cdef extern from "bunruija/keyed_vector.h" namespace "bunruija":
    cdef cppclass CppPretrainedVectorProcessor "bunruija::PretrainedVectorProcessor":
        CppPretrainedVectorProcessor() except +
        CppStatus convert(char *, char *)
        CppStatus load(string)
        CppStatus query(string, vector[float] *)
        CppStatus batch_query(vector[string], unordered_map[string, vector[float]] *)
        CppStatus contains(string, bool *)
        int dim()
        int length()


cdef class Status:
    cdef CppStatus *thisptr

    def __cinit__(self, code=0, message=""):
        cdef string message_ = string(bytes(message))
        self.thisptr = new CppStatus(code, message_)

    def __repr__(self):
        msg = self.thisptr.status_message.decode('utf-8')
        return f'Status(status_code={self.thisptr.status_code}, status_message=\"{msg}\")'


@cython.auto_pickle(True)
cdef class PretrainedVectorProcessor:
    cdef CppPretrainedVectorProcessor *thisptr

    def __init__(self):
        self.thisptr = new CppPretrainedVectorProcessor()

    def convert(self, db_file, input_file):
        status_ = self.thisptr.convert(db_file.encode('utf-8'), input_file.encode('utf-8'))
        status = Status(status_.status_code, status_.status_message)
        return status

    def load(self, input_file):
        cdef string input_file_ = string(bytes(input_file.encode('utf-8')))
        status_ = self.thisptr.load(input_file_)
        status = Status(status_.status_code, status_.status_message)
        return status

    @lru_cache(maxsize=0)
    def query(self, word):
        cdef string word_ = string(bytes(word.encode('utf-8')))
        cdef vector[float] vec_;
        status_ = self.thisptr.query(word_, &vec_)
        status = Status(status_.status_code, status_.status_message)

        vec = np.asarray(vec_)
        return vec, status

    @lru_cache(maxsize=0)
    def batch_query(self, words):
        cdef unordered_map[string, vector[float]] vec_;
        status_ = self.thisptr.batch_query([bytes(w.encode("utf-8")) for w in words], &vec_)
        status = Status(status_.status_code, status_.status_message)

        vecs = []
        for word in words:
            v = vec_[bytes(word.encode("utf-8"))]
            if len(v) == 0:
                v = [0 for _ in range(self.emb_dim)]
            vecs.append(v)
#             print(word, "shape", len(v), type(v), type(v[0]))
        vecs = np.asarray(vecs)
#         print("shape", vecs.shape)

        return vecs, status

    def __contains__(self, word):
        cdef string word_ = string(bytes(word.encode('utf-8')))
        cdef bool out_;
        status_ = self.thisptr.contains(word_, &out_)
        status = Status(status_.status_code, status_.status_message)
        return out_

    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)

    def __reduce_ex__(self, proto):
        func = PretrainedVectorProcessor
        args = ()
        state = self.__getstate__()
        listitems = None
        dictitems = None
        return (func, args, state, listitems, dictitems)

    @property
    def emb_dim(self):
        return self.thisptr.dim()

    @property
    def length(self):
        return self.thisptr.length()
