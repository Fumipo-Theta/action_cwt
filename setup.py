# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='action_cwt',
    version='1.0.0',
    description='Wrapper of pycwt',
    long_description=readme,
    author='Fumitoshi Morisato',
    author_email='fmorisato@gmail.com',
    url='https://github.com/Fumipo-Theta/action_cwt',
    install_requires=[
        'numpy', 'matplotlib', 'pandas', 'pycwt', 'matpos==1.0.0', 'func_helper==1.0.0'],
    dependency_links=[
        'git+https://github.com/Fumipo-Theta/matpos.git@master#egg=matpos-1.0.0',
        'git+https://github.com/Fumipo-Theta/func_helper.git@master#egg=func_helper-1.0.0'],
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
