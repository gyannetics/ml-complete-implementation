from setuptools import setup, find_packages
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    '''
    Returns a list of package requirements
    '''
    requirements = []
    with open(file_path, 'r') as requirements:
        requirements = [pkg.strip() for pkg in requirements.readlines()]
    
    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)

setup(
    name='ML-Complete-Implementation',
    version='0.0.1',
    author='Santosh KV',
    author_email='theaimonk24@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)