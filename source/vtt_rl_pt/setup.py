"""Installation script for the VTT_RL_PT extension."""

from setuptools import setup, find_packages

setup(
    name="vtt_rl_pt",
    packages=find_packages(),
    author="",
    maintainer="",
    url="",
    version="0.0.0",
    description="Procedural Trajectory (PT) extension for Isaac Lab 2.0.1.",
    keywords=["isaaclab", "robotics", "rl", "trajectory", "polishing"],
    install_requires=["numpy", "torch", "gymnasium", "h5py", "pyyaml"],
    license="MIT",
    include_package_data=True,
    package_data={
        "robo_pp_fixed": [
            "assets/*.usda",
            "agents/*.yaml",
            "cfg/*.yaml",
            "data/*.json",
            "data/*.yaml",
        ],
    },
    python_requires=">=3.10",
    zip_safe=False,
)
