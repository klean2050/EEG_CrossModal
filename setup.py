import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="EEG_CrossModal_klean2050",
    version="0.0.1",
    author="Kleanthis Avramidis",
    author_email="k.avramidis@windowslive.com",
    description="ICASSP 2022 Submission",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/klean2050/EEG_CrossModal",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
