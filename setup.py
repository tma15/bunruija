from setuptools import setup, find_packages

setup(
    name='bunruija',
    version='0.0.0',
    packages=find_packages(),
    install_requires=[
        'mecab-python3==0.996.5',
        'pymagnitude',
        'pyyaml',
        'torch',
        'scikit-learn',
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
