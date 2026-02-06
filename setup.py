from setuptools import setup, find_packages
import os
from glob import glob

setup(
    name='RT-COSMIK',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    scripts=[os.path.join('scripts', f) for f in os.listdir('scripts') if f.endswith('.py')],
    
    # Include settings.py as package data
    data_files=[
        ('share/RT-COSMIK', ['settings.py']),  # For installed version
        (os.path.join('share', 'RT-COSMIK'), glob('config/*')),
        (os.path.join('share', 'RT-COSMIK'), glob('launch/*')),
    ],
    
    install_requires=[
        'numpy',
        'opencv-python',
        'dataclasses; python_version<"3.7"',
    ],
)