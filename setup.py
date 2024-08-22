import os
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist


class f2py_Extension(Extension):
    def __init__(self, name, sourcedirs):
        Extension.__init__(self, name, sources=[])
        self.sourcedirs = [os.path.abspath(sourcedir) for sourcedir in sourcedirs]
        self.dirs = sourcedirs


class f2py_Build(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)
        super().run()

    def build_extension(self, ext):
        for ind, to_compile in enumerate(ext.sourcedirs):
            module_loc = os.path.split(ext.dirs[ind])[0]
            module_name = os.path.split(to_compile)[1].split('.')[0]
            command = f'cd {module_loc} && f2py -c {to_compile} -m {module_name}'
            print(f'Executing: {command}')
            os.system(command)


class CustomBuildPy(build_py):
    def run(self):
        self.run_command('build_ext')
        super().run()


class CustomSDist(sdist):
    def run(self):
        self.run_command('build_ext')
        super().run()


setup(
    name='pyadps',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'streamlit',
        'numpy',
        'matplotlib',
        'scipy',
        'wmm2020',
        'cmake',
        'pandas',
        'netCDF4',
        'plotly',
        'plotly-resampler',
        'meson'
    ],
    ext_modules=[f2py_Extension('pyadps.utils.fortreadrdi', ['pyadps/utils/fortreadrdi.v0.7.f95'])],
    cmdclass=dict(
        build_ext=f2py_Build,
        build_py=CustomBuildPy,
        sdist=CustomSDist,
    ),
    entry_points={
        'console_scripts': [
            'run-pyadps = pyadps.__main__:main',
        ],
    },
    author="amol",
    author_email="velip09@gmail.com",
    description="A package for ADPS with Streamlit web interface",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.6',
)

