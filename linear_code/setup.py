from setuptools import setup

install_requires = [
    "jax",
    "optax",
    "numpy",
    "pandas",
    "dm-haiku",
    "dm-env",
    "rlax",
    "chex",
    "absl-py",
]

setup(
    name="linear-representation-learning",
    version="",
    packages=["replearn"],
    install_requires=install_requires,
    url="",
    license="",
    author="Clement Gehring",
    author_email="",
    description="",
)
