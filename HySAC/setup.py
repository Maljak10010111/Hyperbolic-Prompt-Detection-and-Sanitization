from setuptools import setup, find_packages
from pathlib import Path

# Read the README file for the long description
long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
    name="hysac",
    version="0.1.0",
    description="Hybrid Secure and Adversarially Robust Computing (HySAC)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",  # Replace with your name
    author_email="your.email@example.com",  # Replace with your email
    packages=find_packages(),  # Automatically include all packages
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "scipy",
        "torch",
    ],
)
