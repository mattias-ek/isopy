import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="isopy",
    version="0.0.1",
    author="Mattias Ek",
    author_email="mattias.ek@erdw.ethz.ch",
    description="A python package for isotope geo/cosmochemistry",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mattias-ek/isopy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
    ],
)