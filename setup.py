from setuptools import setup
from setuptools import find_packages

setup(
    name="summon-the-demon",
    version="1.0",
    description="Cthulhu: Deep Learning for demons.",
    long_description="Cthulhu: Deep Learning for demons.",
    url="https://github.com/maxpumperla/cthulhu",
    download_url="https://github.com/maxpumperla/cthulhu/tarball/1.0",
    author="Max Pumperla",
    author_email="max.pumperla@googlemail.com",
    install_requires=[
        "keras>=2.0.4",
    ],
    packages=find_packages(),
    license="MIT",
    zip_safe=False,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Environment :: Console",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
    ],
)
