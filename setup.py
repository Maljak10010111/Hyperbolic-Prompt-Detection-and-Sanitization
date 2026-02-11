from setuptools import setup, find_packages
setup(
    name="HySAC",
    version="0.1",
    packages=find_packages(),  # This will find all packages
    include_package_data=True,
    install_requires=[
    ],
    python_requires='>=3.7',
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
