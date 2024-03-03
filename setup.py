
import os.path
import codecs
from setuptools import setup, find_packages


setup(
    name="HamGNN",
    version='0.1.0',
    description="Hamiltonian prediction via Graph Neural Network",
    download_url="",
    author="Yang Zhong",
    python_requires=">=3.9",
    packages=find_packages(),
    package_dir={},
    package_data={'': ['*.npz', '*.json'],},
    entry_points={
        "console_scripts": [
            "HamGNN = HamGNN.main:HamGNN",
            "band_cal = utils_openmx.band_cal:main",
            "graph_data_gen = utils_openmx.graph_data_gen:main"
        ]
    },
    install_requires=[
        "numpy",
        "torch==1.11.0",
        "torch_scatter==2.0.9",
        "torch_sparse==0.6.15",
        "torch_geometric",
        "aiohttp==3.9.3",
        "pytorch_lightning==1.9.0",
        "mpmath==1.3.0",
        "torch_runstats",
        "e3nn",
        "pymatgen",
        "ase",
        "tqdm",
        "tensorboard",
        "natsort",
        "easydict",
        "numba",
        "pillow==9.5.0"
    ],
    license="MIT",
    license_files="LICENSE",
    zip_safe=False,
)
