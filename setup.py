import setuptools
from os import path
import hlapred

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
# with open(path.join(here, 'README.md')) as f:
#     long_description = f.read()

if __name__ == "__main__":
    setuptools.setup(
        name='HLAIIPred',
        version=hlapred.__version__,
        author='Mojtaba Haghighatlari',
        author_email='mojtaba.haghighatlari@pfizer.com',
        project_urls={
            'Source': 'https://github.com/pfizer-rd/HLAIIPred',
        },
        description=
        "A Python library that provides tools and models for deep learning of HLA class II epitopes.",
        # long_description=long_description,
        # long_description_content_type="text/markdown",
        scripts=['cli/HLAIIPred'],
        keywords=[
            'Machine Learning', 'HLAII', 'MHC', 'Deep Learning',
        ],
        license='MIT',
        packages=setuptools.find_packages(),
        include_package_data=True,
        classifiers=[
            'Development Status :: 4 - Beta',
            'Natural Language :: English',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 3',
        ],
        zip_safe=False,
    )