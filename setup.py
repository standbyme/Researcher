from setuptools import setup, find_packages

setup(
    name='cycleresearcher',
    version='0.1.0',
    description='CycleResearcher, A LLM for autoresearcher',
    author='xxx',
    author_email='xxx',
    url='xxx',
    packages=find_packages(where='./cycleresearcher'),
    package_dir={'': './cycleresearcher'},
    install_requires=[
        'numpy',
        'pandas',
        'torch',
        'transformers',
        'scikit-learn',
        'tqdm',
        'openai',
        'matplotlib',
        # Add other dependencies as needed
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)