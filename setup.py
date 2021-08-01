import setuptools


def setup_package():
  long_description = "hyperformer"
  with open('requirements.txt') as f:
      required = f.read().splitlines()
  setuptools.setup(
      name='hyperformer',
      version='0.0.1',
      description='HyperFormer',
      long_description=long_description,
      long_description_content_type='text/markdown',
      license='MIT License',
      packages=setuptools.find_packages(
          exclude=['docs', 'tests', 'scripts', 'examples']),
      dependency_links=[
          'https://download.pytorch.org/whl/torch_stable.html',
      ],
      install_requires = required,
      classifiers=[
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.7',
      ],
      keywords='text nlp machinelearning',
  )


if __name__ == '__main__':
  setup_package()
