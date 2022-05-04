try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(name='schism',
      version="0.0.1",
      use_scm_version=True,
      setup_requires=['setuptools_scm'],
      description="Immersed boundary tools for Devito codes",
      long_description="""
      Schism is a package intended to facilitate immersed boundary
      implementation in finite-difference codes written with Devito.
      """,
      url='http://www.devitoproject.org/',
      author="Imperial College London",
      author_email='ec815@ic.ac.uk.ac.uk',
      license='MIT',
      packages=['schism'],
      install_requires=['numpy', 'sympy'])
