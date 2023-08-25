#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 6/12/23 3:17 PM
# @Author  : zhangchao
# @File    : setup.py
# @Email   : zhangchao5@genomics.cn
import setuptools
from wheel.bdist_wheel import bdist_wheel

__version__ = "0.1.4"


class BDistWheel(bdist_wheel):
    def get_tag(self):
        return (self.python_tag, "none", "any")


cmdclass = {
    "bdist_wheel": BDistWheel,
}

requirements = open("requirements.txt").readline()

setuptools.setup(
    name="spatialign",
    version=__version__,
    author="zhangchao",
    author_email="1623804006@qq.com",
    description="spatiAlign: An Unsupervised Contrastive Learning Model for Data Integration of Spatially Resolved Transcriptomics",
    long_description_content_type="text/markdown",
    url="https://github.com/zhangchao162/Spatialign.git",
    packages=setuptools.find_packages(),
    package_data={'': ["*.so"]},
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    include_package_data=True,
    cmdclass=cmdclass
)
