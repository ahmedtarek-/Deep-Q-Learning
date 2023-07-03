from setuptools import setup

# with open("README.md", "r") as fh:
#     long_description = fh.read()

setup(name='gym_grid',
      version='0.0.1',
      author='Robert Tjrako Lange',
      author_email='robert.t.lange@tu-berlin.de',
      license='MIT',
      description="An OpenAI Gym Environment for Custom Gridworlds.",
      #long_description=long_description,
      long_description_content_type="text/markdown",
      url="",
      install_requires=['numpy', 'gym', 'matplotlib']
      )
