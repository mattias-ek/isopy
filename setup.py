import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="isopy",
    version='0.7.4',
    packages=setuptools.find_packages(include=['isopy', 'isopy.*']),
    install_requires=['numpy>=>1.20',
                      'pyperclip>=1.8',
                      'openpyxl>=3',
                      'matplotlib>=3.3',
                      'scipy>=1.6',
                      'chardet>=4',
                      'tabulate>=0.8'],
    python_requires = '>=3.9',
    setup_requires=['pytest-runner'],
    tests_require=['pytest', 'pytest-mpl'],
    include_package_data=True,

    author="Mattias Ek",
    author_email="mattias.ek@erdw.ethz.ch",
    description="A Python 3 library for data processing in isotope geo/cosmochemistry.",
    long_description=long_description,
    keywords="array isotope geochemistry cosmochemisty geology icpms icp-ms",
    long_description_content_type="text/markdown",
    url="https://github.com/mattias-ek/isopy",

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education"
    ],
)
