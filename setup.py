from setuptools import setup, find_packages

from recalibration import __version__

setup(name='recalibration',
      version=__version__,
      description='Python library for recalibrating ms data',
      url='',
      author='Andrew Palmer, Alexandrov Team, EMBL',
      packages=find_packages(),
      install_requires=[
          'elasticsearch==5.4.0',
          'elasticsearch_dsl==5.3.0',
          'pandas',
          'plotly',
          'numpy',
          'pyyaml',
          'psycopg2',
          'matplotlib',
          'pyMSpec',
          'pyImagingMSpec',
          'boto3',
          'joblib',
          'scipy',
          'networkx'
      ])
