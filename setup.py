"""
    * @file: setup.py
    * @version: v1.0.0
    * @author: Zhiyu Zhang
    * @desc: setup for dasQt
    * @date: 2023-07-25 12:58:34
    * @Email: erbiaoger@gmail.com
    * @url: erbiaoger.site
"""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dasQt",
    version="1.0.0",
    author="Zhiyu Zhang",
    author_email="erbiaoger@gmail.com",
    description="dasQt - open source DAS data processing software",
    entry_points={'console_scripts': ['dasQt = dasQt.__main__:main']},
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/erbiaoger/dasQt",
    packages=['dasQt'],
    # package_data={'dasQt': ['exampledata/GSSI/*.DZT',]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['numpy','scipy','matplotlib','pyqt6','h5py','numba']
)
