from setuptools import setup, find_packages
setup(
    name="HySAC",
    version="0.1",
    packages=find_packages(),  # This will find all packages
    include_package_data=True,
    install_requires=[
        # List dependencies here, for example:
        # "torch>=1.12.0",
        # "numpy>=1.21.0"
    ],
    python_requires='>=3.7',
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)