import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="quickshow",
    version="0.1.9",
    author="parkminwoo",
    author_email="parkminwoo1991@gmail.com",
    description="Quick-Show provides simply but powerful insight plots",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DSDanielPark/quick-show",
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    install_requires=[
        "seaborn",
        "numpy",
        "pandas",
        "matplotlib",
        "scikit-learn"
    ])