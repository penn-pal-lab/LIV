import os
import sys
from setuptools import setup, find_packages

if sys.version_info.major != 3:
    print("This Python is only compatible with Python 3, but you are running "
          "Python {}. The installation will likely fail.".format(sys.version_info.major))
    
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='liv',
    version='0.0.0',
    packages=find_packages(),
    description='LIV: Language-Image Representations and Rewards for Robotic Control',
    long_description=read('README.md'),
    author='Jason Ma',
    install_requires=[
        'torch<=1.13.1',
        'torchvision>=0.8.2',
        'omegaconf==2.1.1',
        'hydra-core==1.1.1',
        'pillow==9.0.1',
        'opencv-python',
        'matplotlib',
        'flatten_dict',
        'gdown', 
        'huggingface_hub',
        'tabulate',
        'pandas',
        'scipy',
        'scikit-learn',
        'scikit-video',
        'transforms3d',
        'moviepy',
        'termcolor',
        'wandb' 
    ]
)
