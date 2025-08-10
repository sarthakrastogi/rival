from setuptools import setup, find_packages

with open("README.MD", "r") as f:
    readme_content = f.read()

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

setup(
    name="rival-ai",
    version="0.1.7",
    description="A library for testing and protecting AI agent safety",
    long_description=readme_content,
    long_description_content_type="text/markdown",
    author="Sarthak Rastogi",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=install_requires,
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
