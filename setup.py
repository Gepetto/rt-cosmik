from setuptools import setup, find_packages
import os
from glob import glob

setup(
    name='rtcosmik',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    
    # Install executable scripts
    scripts=[f'scripts/{f}' for f in os.listdir('scripts') if f.endswith('.py')],
    
    # ROS data files
    data_files=[
        (os.path.join('share', 'rtcosmik'), ['package.xml']),
        (os.path.join('share', 'rtcosmik/launch'), glob('launch/*')),
        (os.path.join('share', 'rtcosmik/config'), glob('config/*')),
    ],
    
    install_requires=[
        'numpy',
        'opencv-python',
        # Add other ROS/python dependencies from package.xml
    ],
)