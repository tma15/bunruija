from setuptools import Extension  # type: ignore
from distutils.command.build_ext import build_ext

from Cython.Build import cythonize  # type: ignore


def build(setup_kwargs):
    ext_modules = cythonize(
        [
            Extension(
                "bunruija.modules.vector_processor",
                sources=(
                    ["bunruija/modules/vector_processor.pyx"]
                    + ["csrc/keyed_vector.cc", "csrc/string_util.cc"]
                ),
                libraries=["sqlite3"],
                include_dirs=["include"],
                extra_compile_args=["-std=c++11"],
                extra_link_args=["-std=c++11"],
            ),
        ]
    )

    setup_kwargs.update(
        {"ext_modules": ext_modules, "cmdclass": {"build_ext": build_ext}}
    )
