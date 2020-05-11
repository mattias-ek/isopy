import setuptools
import isopy

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="isopy",
    version=isopy.__version__,
    packages=setuptools.find_packages(include=['isopy', 'isopy.*']),
    install_requires=['numpy', 'tables', 'pyperclip', 'xlrd', 'matplotlib'],

    setup_requires=['pytest-runner'],
    tests_require=['pytest'],

    package_data = {
        "": ["*.txt", "*.rst", "*.csv", "*.xlsx"]},

    author="Mattias Ek",
    author_email="mattias.ek@bristol.ac.uk",
    description="A python package for data processing in geo/cosmochemistry",
    long_description=long_description,
    keywords="array isotope geochemistry cosmochemisty",
    long_description_content_type="text/markdown",
    url="https://github.com/mattias-ek/isopy",

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha"
    ],
)