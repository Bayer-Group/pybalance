import setuptools

__version__ = "0.1.7"

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("environments/requirements.txt") as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="pybalance",
    version=__version__,
    author="Stephen Privitera",
    author_email="stephen.privitera@bayer.com",
    description="Population Matching",
    long_description=long_description,
    install_requires=requirements,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    package_data={"pybalance": ["sim/data/*parquet", "sim/data/*csv"]},
)
