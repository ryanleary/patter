try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension

config = {
    'description': 'Patter - Speech Recognition Toolkit',
    'author': 'Ryan Leary',
    'author_email': 'ryanleary@gmail.com',
    'version': '0.1',
    'install_requires': ['torch', 'python-levenshtein', 'librosa', 'tqdm'],
    'packages': ['patter', 'patter.models', 'patter.util'],
    'name': 'patter',
    'scripts': ['scripts/patter-train', 'scripts/patter-test', 'scripts/patter-serve', 'scripts/patter-client'],
    'test_suite': 'nose.collector',
    'tests_require': ['nose']
}

setup(**config)
