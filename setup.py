import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="evonp", # Replace with your own username
    version="1.0.0",
    author="Raneem Qaddoura, Hossam Faris, and Ibrahim Aljarah",
    author_email="raneem.qaddoura@gmail.com",
    description="An efficient evolutionary algorithm with a nearest neighbor search technique for clustering analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RaneemQaddoura/evonp",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License 2.0",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
