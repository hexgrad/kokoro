from setuptools import setup, find_packages

setup(
    name='kokoro',
    version='0.7.16',
    packages=find_packages(),
    install_requires=[
        'huggingface_hub',
        'loguru',
        'misaki[en]>=0.7.16',
        'numpy==1.26.4',
        'scipy',
        'torch',
        'transformers',
    ],
    python_requires='>=3.7',
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