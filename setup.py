#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 6/12/23 3:17 PM
# @Author  : zhangchao
# @File    : setup.py
# @Email   : zhangchao5@genomics.cn
import setuptools

__version__ = "0.0.1"

requirements = open("requirements.txt").readline()


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="spatialign",
    version=__version__,
    author="zhangchao",
    author_email="1623804006@qq.com",
    description="spatialign: a batch alignment method for spatial transcriptomics via spatial embedding and unsupervised cross-domain adaptation contrastive learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zhangchao162/Spatialign.git",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)


