from setuptools import Extension, setup, find_packages
from Cython.Build import cythonize


setup(
    name='bunruija',
    version='0.0.0',
    packages=find_packages(),
    ext_modules=cythonize([
        Extension(
            'bunruija.modules.vector_processor',
            sources=['bunruija/modules/vector_processor.pyx'],
            libraries=['sqlite3']
        )
    ]),
    install_requires=[
        'lightgbm',
        'mecab-python3==0.996.5',
        'pyyaml',
        'torch>=1.6.0',
        'transformers>=3.5.1',
        'scikit-learn>=0.23.2',
        'unidic-lite',
    ],
    entry_points={
        'console_scripts': [
            'bunruija-evaluate = bunruija_cli.evaluate:cli_main',
            'bunruija-predict = bunruija_cli.predict:cli_main',
            'bunruija-preprocess = bunruija_cli.preprocess:cli_main',
            'bunruija-train = bunruija_cli.train:cli_main',
        ],
    },
)
