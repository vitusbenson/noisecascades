from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()


install_requires = [
    "numpy",
    "xarray",
    "zarr",
    "pyyaml",
    "numcodecs",
    "tqdm",
    "scipy",
    "numba",
    "fire"
]


setup(name='noisecascades', 
        version='0.0.1',
        url = 'https://github.com/vitusbenson/noisecascades',
        author = 'Vitus Benson',
        author_email = 'vbenson@bgc-jena.mpg.de',
        long_description=long_description,
        long_description_content_type="text/markdown",
        classifiers=[
                "Intended Audience :: Science/Research",
                "License :: MIT",
                "Programming Language :: Python :: 3"
                 ],
        packages=find_packages(),
        install_requires=install_requires,
        )
