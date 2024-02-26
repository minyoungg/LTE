import sys
import setuptools


setuptools.setup(
    name="lte",
    version="0.0.1",
    author="Minyoung Huh",
    author_email="minhuh@mit.edu",
    description=f"PyTorch LTE",
    url="git@github.com:minyoungg/lte.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
    )
