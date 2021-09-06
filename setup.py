import setuptools
import os

# Get requirements
thelibFolder = os.path.dirname(os.path.realpath(__file__))
requirementPath = thelibFolder + '/requirements.txt'
install_requires = []
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bifrost",
    version="0.0.4",
    author="Michael Reefe",
    author_email="michael.reefe8@gmail.com",
    description="Correct, normalize, coadd, and stack spectra together.",
    long_description=long_description,
    long_description_content_type="text/x-md",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=install_requires,
    url="https://github.com/Michael-Reefe/bifrost",
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Operating System :: Unix"
    ],
    python_requires='>=3.7'
)
