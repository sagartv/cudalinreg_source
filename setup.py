import os
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import subprocess

# Define the CUDA extension
cuda_extension = Extension(
    'cudalinreg._linalg', # name of the output module: cudalinreg/_linalg.so
    sources=['cudalinreg/linalg_cuda.cu'],
    libraries=['cudart'],
    library_dirs=[os.path.join(os.environ.get('CUDA_HOME', '/usr/local/cuda'), 'lib64')],
    extra_compile_args={'cxx': [], 'nvcc': ['-arch=sm_75']} # Use sm_87 for Orin, sm_75 for Colab T4, etc.
)

class CudaBuildExt(build_ext):
    """Custom build extension to handle CUDA files."""
    def build_extensions(self):
        # Find the CUDA compiler
        nvcc = os.path.join(os.environ.get('CUDA_HOME', '/usr/local/cuda'), 'bin', 'nvcc')
        if not os.path.exists(nvcc):
            raise FileNotFoundError("The nvcc compiler could not be found. Make sure CUDA_HOME is set.")

        for ext in self.extensions:
            # Create the output directory if it doesn't exist
            os.makedirs(os.path.dirname(self.get_ext_fullpath(ext.name)), exist_ok=True)
            
            # Compile the CUDA source file into a shared object
            cmd = [
                nvcc,
                '--shared',
                '-Xcompiler', '-fPIC',
                '-o', self.get_ext_fullpath(ext.name)
            ] + ext.sources + ext.extra_compile_args['nvcc']
            
            print(f"Running command: {' '.join(cmd)}")
            subprocess.check_call(cmd)

setup(
    name='cudalinreg',
    version='0.1.0',
    description='A Python library for CUDA-accelerated linear regression.',
    author='You',
    packages=['cudalinreg'],
    ext_modules=[cuda_extension],
    cmdclass={'build_ext': CudaBuildExt}
) 