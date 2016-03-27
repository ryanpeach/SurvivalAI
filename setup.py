from distutils.core import setup

setup(name='SurvivalAI',
      version='0',
      description='An AI for Minecraft and Minetest',
      author='Ryan Peach',
      author_email='ryan.peach@outlook.com',
      url='http://www.github.com/ryanpeach/SurvivalAI',
      packages=['numpy', 'scipy', 'pandas', 'https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.7.1-cp34-none-linux_x86_64.whl'],
     )