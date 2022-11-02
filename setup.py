from pathlib import Path
from setuptools import setup, find_packages

description = ['LaMAR: Benchmarking Localization and Mapping for Augmented Reality']

root = Path(__file__).parent
with open(str(root / 'README.md'), 'r', encoding='utf-8') as f:
    readme = f.read()

setup(
    name='lamar',
    version='1.0',
    packages=find_packages(),
    python_requires='>=3.8',
    author='Microsoft',
    description=description,
    long_description=readme,
    long_description_content_type="text/markdown",
    url='https://github.com/microsoft/lamar-benchmark',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
