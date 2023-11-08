"""Setup file for pip installation."""

import setuptools

REQUIRED_PACKAGES = ['absl-py', 'numpy', 'pandas', 'scikit-learn', 'scipy']

setuptools.setup(
    name='vet',
    version='1.0',
    description='A library to measure ML model result variances.',
    long_description='',
    url='http://github.com/google-research/vet/',
    author='Chris Welty',
    author_email='welty@google.com',
    # Contained modules and scripts.
    packages=setuptools.find_packages(),
    install_requires=REQUIRED_PACKAGES,
    python_requires='>=3.10.0',
    zip_safe=False,
    license='Apache 2.0',
    keywords='machine learning model variance metrics',
)
