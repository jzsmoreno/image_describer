from pathlib import Path

import setuptools

# Parse the requirements.txt file
with open("requirements.txt", "r") as f:
    install_requires = f.read().splitlines()

with open("README.md", "r") as fh:
    long_description = fh.read()

about = {}
ROOT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = ROOT_DIR / "image_describer"
with open(PACKAGE_DIR / "VERSION") as f:
    _version = f.read().strip()
    about["__version__"] = _version

setuptools.setup(
    name="image_describer",
    version=about["__version__"],
    packages=setuptools.find_packages(),
    install_requires=install_requires,
)
