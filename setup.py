#!/usr/bin/env python
# coding=utf-8

from setuptools import setup, find_packages

setup(
    name='little_dinosaur',
    version='0.0.3',
    description=(
        'utils for nlp'
    ),
    author='liyuhao',
    author_email='1241225413@qq.com',
    license='Apache License 2.0',
    packages=find_packages(),
    url='https://github.com/hgliyuhao',
    install_requires=[
        'fairies',
        'bert4keras',
        'GitPython',
        'numpy'
    ],
)
