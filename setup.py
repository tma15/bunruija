from setuptools import setup, find_packages

setup(
    name='bunruija',
    version='0.0.0',
    packages=find_packages(),
    install_requires=[
        'mecab-python3',
        'pyyaml',
    ],
    entry_points={
        'console_scripts': [
            'bunruija-evaluate = bunruija_cli.evaluate:main',
            'bunruija-predict = bunruija_cli.predict:main',
            'bunruija-preprocess = bunruija_cli.preprocess:main',
            'bunruija-train = bunruija_cli.train:main',
        ],
    },
)
