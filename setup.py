from setuptools import setup, find_packages

setup(
    name='kokoro',
    version='0.3.1',
    packages=find_packages(),
    install_requires=[
        'huggingface_hub',
        'misaki[en]>=0.6.1',
        'numpy==1.26.4',
        'scipy',
        'torch',
        'transformers',
    ],
    python_requires='>=3.6',
    author='hexgrad',
    author_email='hello@hexgrad.com',
    description='TTS',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/hexgrad/kokoro',
    license='Apache 2.0',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
)