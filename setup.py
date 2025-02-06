from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT='-e .'

def get_requirements(file_path:str)->List[str]:
    '''
    Returns requirement array!
    '''
    requirements=[] 
    
    with open(file_path, 'r') as file:
        requirements=file.readlines()
        requirements=[req.replace("\n","") for req in requirements]
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements

setup(
    name="mlproject",
    version='0.0.1',
    author="Ayush",
    author_email="ayushjaswal4543@gmail.com",
    packages=find_packages(),
    requirements=get_requirements('requirements.txt')
)
