import setuptools
import os
import re

# Get requirements
thelibFolder = os.path.dirname(os.path.realpath(__file__))
requirementPath = thelibFolder + '/requirements.txt'
install_requires = []
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()

with open("README.md", "r") as fh:
    long_description = fh.read()

search = re.search(r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]', open('bifrost/__init__.py').read())
version = search.group(1)

setuptools.setup(
    name="astro-bifrost",
    version=version,
    author="Michael Reefe",
    author_email="michael.reefe8@gmail.com",
    description="Correct, normalize, coadd, and stack spectra together.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=install_requires,
    url="https://github.com/Michael-Reefe/bifrost",
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Operating System :: Unix"
    ],
    # entry_points={
    #     'console_scripts': ['bifrost=bifrost.cli:main'],
    # },
    python_requires='>=3.7'
)
