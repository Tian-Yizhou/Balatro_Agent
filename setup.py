from setuptools import setup, find_packages

# NOTE: Preferred setup method is via conda environments:
#   conda env create -f environment.yml       # full project (balatro-agent)
#   conda env create -f environment_gym.yml   # environment only (balatro-gym)
# See README.md for details.

setup(
    name="balatro_gym",
    version="0.1.0",
    description="A Gymnasium-compatible card game environment inspired by Balatro",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "gymnasium>=0.29.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "recording": ["pyarrow>=12.0"],
        "agent": ["ray[rllib]>=2.10.0", "torch>=2.0.0"],
        "dev": ["pytest>=7.0.0"],
        "all": [
            "pyarrow>=12.0",
            "ray[rllib]>=2.10.0",
            "torch>=2.0.0",
            "pytest>=7.0.0",
        ],
    },
    entry_points={
        "gymnasium.envs": ["balatro_gym = balatro_gym"],
    },
)
