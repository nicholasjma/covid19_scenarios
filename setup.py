from setuptools import setup, find_packages
import versioneer

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as fh:
    requirements = fh.read().splitlines()

setup(
    name="covid19_chime_county",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    license="Copyright Cerner Co. 2020",
    long_Description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=requirements,
)
