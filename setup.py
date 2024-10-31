from setuptools import setup, find_packages

setup(
    name="pcs956_project",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "asttokens>=2.4.1",
        "contourpy>=1.3.0",
        "cycler>=0.12.1",
        "joblib>=1.4.2",
        "matplotlib>=3.9.2",
        "numpy>=2.1.2",
        "packaging>=24.1",
        "pandas>=2.2.3",
        "pyarrow>=17.0.0",
        "pytest>=8.3.3",
        "scikit-learn>=1.5.2",
        "scipy>=1.14.1",
        "seaborn>=0.13.2",
    ],
    extras_require={
        "dev": [
            "ipykernel>=6.29.5",
            "jupyter_client>=8.6.3",
            "jupyter_core>=5.7.2",
            "ipython>=8.28.0",
            "debugpy>=1.8.7",
        ]
    },
    python_requires=">=3.8",
)
