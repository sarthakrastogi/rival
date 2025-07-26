from setuptools import setup, find_packages

with open("README.MD", "r") as f:
    readme_content = f.read()

setup(
    name="rival-ai",
    version="0.1.5",
    description="A library for testing and protecting AI agent safety",
    long_description=readme_content,
    long_description_content_type="text/markdown",
    author="Sarthak Rastogi",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pydantic>=2.0.0",
        "litellm>=1.66.1",
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "sentence-transformers>=2.2.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "huggingface_hub>=0.16.0",
        "datasets>=2.14.0",
        "accelerate>=0.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
        ]
    },
    python_requires=">=3.8",
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
