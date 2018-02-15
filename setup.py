# Setup script for installing techbubbleiotjumpwaymqtt#
# Author:   Adam Milton-Barker <adammiltonbarker@gmail.com>#
# Copyright (C) 2016 - 2017 TechBubble Technologies Limited
# For license information, see LICENSE.txt

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='InceptionFlow',
    version="0.5.0",
    author='Adam Milton-Barker',
    author_email='adammiltonbarker@gmail.com',
    url='https://github.com/AdamMiltonBarker/InceptionFlow',
    license='',
    description='Object & Facial Recognition With IoT Communication. Using Tensorflow Inception V3 & TechBubble IoT JumpWay',
    packages=['InceptionFlow'],
    install_requires=[
        "techbubbleiotjumpwaymqtt >= 0.3.9"
    ],
    classifiers=[],
)