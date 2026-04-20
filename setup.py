from setuptools import setup, find_packages

setup(
    name="balatro_gym",
    version="0.1.0",
    description="A Gymnasium-compatible card game environment inspired by Balatro",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "gymnasium>=0.29.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "recording": ["pyarrow>=12.0"],
    },
    entry_points={
        "gymnasium.envs": ["balatro_gym = balatro_gym"],
    },
)
