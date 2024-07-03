from setuptools import setup, find_packages

setup(
    name="text_comparison",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        "spacy>=3.0.0",
        "numpy>=1.20.0",
        "sentence-transformers>=2.0.0"
    ],
    author="Pragadeshwar Vishnu",
    author_email="mkpvishnu@gmail.com",
    description="A library for comparing texts across various tasks",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/mkpvishnu/text-similarity",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)