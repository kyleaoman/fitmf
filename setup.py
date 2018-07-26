from setuptools import setup

setup(
    name='fitmf',
    version='1.0',
    description='Utilities for fitting mass functions.',
    url='',
    author='Kyle Oman',
    author_email='koman@astro.rug.nl',
    license='GNU GPL v3',
    packages=['fitmf'],
    install_requires=['numpy', 'astropy', 'matplotlib', 'emcee', 'corner'],
    include_package_data=True,
    zip_safe=False
)
