try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension
from pathlib import Path

install_requirements = ["torch", "numpy", "python-levenshtein", "librosa", "SoundFile",
                        "tqdm", 'toml', 'tensorboardX', 'marshmallow==2.15.1']
test_requirements = ["nose"]

script_root = Path("scripts")
packages = ["patter", "patter.config", "patter.decoder", "patter.models", "patter.util"]
scripts = [
    script_root / "patter-train",
    script_root / "patter-test",
    script_root / "patter-serve",
    script_root / "patter-client",
    script_root / "patter-model",
    script_root / "patter-convert"
]

setup(
    description="Patter - Speech Recognition Toolkit",
    author="Ryan Leary",
    author_email="ryanleary@gmail.com",
    version="0.1",
    install_requires=install_requirements,
    packages=packages,
    name="patter",
    test_suite="nose.collector",
    scripts=[str(p) for p in scripts],
    tests_require=test_requirements)
