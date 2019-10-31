from setuptools import setup, find_packages


def readme():
    with open('README.md') as f:
        return f.read()


dependencies = [
    'numpy>=1.9.2',
    'pandas>=0.16.0',
]

setup(name='econtools',
      version='0.3.0',
      description='Econometrics and other tools',
      long_description=readme(),
      # url=
      author='Daniel Sullivan',
      # author_email=
      packages=find_packages(),
      # install_requires=dependencies,
      tests_require=[
          'pytest',
      ],
      include_package_data=True,        # To copy stuff in `MANIFEST.in`
      package_data={'econtools': ["py.typed"]},
      zip_safe=False,
      # dependency_links=['http://
      license='BSD'
      )
