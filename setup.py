from setuptools import setup, find_packages


def readme():
    with open('README.md') as f:
        return f.read()

setup(name='econtools',
      version='0.0.1',
      description='Econometrics and other tools',
      long_description=readme(),
      # url=
      author='Daniel Sullivan',
      # author_email=
      packages=find_packages(),
      install_requires=[
          'numpy>=1.9.2',
          'pandas>=0.16.0',
          'nose',
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      include_package_data=True,        # To copy stuff in `MANIFEST.in`
      # dependency_links=['http://
      # zip_safe=False
      # license='MIT'
      )
