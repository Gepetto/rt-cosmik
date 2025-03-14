from setuptools import setup, find_packages
import os
from glob import glob

setup(
    name='rtcosmik',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    scripts=[os.path.join('scripts', f) for f in os.listdir('scripts') if f.endswith('.py')],
    
    # Include settings.py as package data
    data_files=[
        ('share/rtcosmik', ['settings.py']),  # For installed version
        (os.path.join('share', 'rtcosmik'), glob('config/*')),
        (os.path.join('share', 'rtcosmik'), glob('launch/*')),
    ],
    
    install_requires=[
        'numpy',
        'opencv-python',
        'dataclasses; python_version<"3.7"',
    ],
)