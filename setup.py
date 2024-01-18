import setuptools
from pybalance import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("environments/requirements.txt") as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="pybalance",
    version=__version__,
    author="IEG Data Science",
    author_email="author@example.com",
    description="Population Matching",
    long_description=long_description,
    install_requires=requirements,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
