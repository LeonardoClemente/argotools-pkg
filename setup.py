import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="argotools",
    version="0.0.3",
    author="Leonardo Clemente",
    author_email="clemclem1991@gmail.com",
    description="Code tools to facilitate Digital Disease Surveillance",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
