from setuptools import setup, find_packages

setup(
    name="lingualens",
    version="0.7.0",
    packages=find_packages(),
    install_requires=[
        "spacy>=3.0.0",
        "numpy>=1.20.0",
        "scikit-learn>=0.24.0",
        "nltk>=3.6.0",
        "sentence-transformers>=2.0.0",
        "openai==0.28.0"
    ],
    author="Pragadeshwar Vishnu",
    author_email="mkpvishnu@example.com",
    description="A library for comparing and analyzing text similarity",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/mkpvishnu/text_compare",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)