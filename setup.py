from setuptools import setup

setup(
    name='hand_decoding',
    url='https://github.com/arthur-pe/hand-decoding.git',
    author='arthur',
    #author_email='',
    packages=['hand_decoding'],
    install_requires=['numpy', 'matplotlib'],
    version='0.1',
    #license='MIT',
    description='Package for decoding hand movement from neural data.',
    long_description=open('README.txt').read(),
)