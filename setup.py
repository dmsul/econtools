from setuptools import setup, find_packages


def readme():
    with open('README.md') as f:
        return f.read()


dependencies = [
    'pandas>=0.16.0',
    'numpy>=1.9.2',
    'scipy',
]

setup(name='econtools',
      version='0.3.2',
      description='Econometrics and other tools',
      long_description=readme(),
      long_description_content_type="text/markdown",
      url=r'http://www.danielmsullivan.com/econtools',
      author='Daniel M. Sullivan',
      # author_email='',
      packages=find_packages(),
      install_requires=dependencies,
      tests_require=[
          'pytest',
      ],
      include_package_data=True,        # To copy stuff in `MANIFEST.in`
      package_data={'econtools': ["py.typed"]},
      zip_safe=False,
      license='BSD',
      python_requires='>=3.6',
      classifiers=[
          'Development Status :: 4 - Beta',
          'License :: OSI Approved :: BSD License',
          'Topic :: Scientific/Engineering :: Mathematics',
          'Topic :: Sociology',
          'Topic :: Scientific/Engineering :: Information Analysis',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Operating System :: OS Independent',
      ]
      )
