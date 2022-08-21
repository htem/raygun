from setuptools import find_packages, setup
setup(
    name='raygun',
    packages=find_packages(include=['raygun']),
    version='0.0.1',
    description='Library for testing and implementing image denoising, enhancement, and segmentation techniques on large, volumetric datasets. Built with Gunpowder, Daisy, PyTorch, and eventually JAX (hopefully).',
    author='Jeff Rhoades',
    author_email='rhoades@g.harvard.edu',
    license='AGPL-3.0',
    requires=[] #TODO
)