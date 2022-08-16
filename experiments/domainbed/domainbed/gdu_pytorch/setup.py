import setuptools

setuptools.setup(
    name='gdu_pytorch',
    version='0.0.2',
    author='András Sass',
    description='PyTorch Implementation of the Gated Domain Unit (GDU) Module for Domain Generalization',
    url='https://github.com/im-ethz/gdu4dg-pytorch',
    license='MIT',
    packages=['toolbox'],
    install_requires=['torch'],
)

