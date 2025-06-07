import setuptools
from importlib.machinery import SourceFileLoader

version = SourceFileLoader(
    'museval-ssir.version', 'museval-ssir/version.py'
).load_module()

with open('README_legacy.md', 'r') as fdesc:
    long_description = fdesc.read()

if __name__ == "__main__":
    setuptools.setup(
        # Name of the project
        name='museval-ssir',

        # Version
        version=version._version,

        # Description
        description='Modified evaluation tools for the SIGSEP MUS database',
        url='https://github.com/MalakiasK/sigsep-mus-eval-ssir',

        # Your contact information
        author='Malakias Kosonen',
        author_email='malakias.kosonen@tuni.fi',

        # License
        license='MIT',

        # Packages in this project
        # find_packages() finds all these automatically for you
        packages=setuptools.find_packages(),

        long_description=long_description,
        long_description_content_type='text/markdown',

        entry_points={
            'console_scripts': [
                'museval=museval.cli:museval',
                'bsseval=museval.cli:bsseval'
            ],
        },
        # Dependencies, this installs the entire Python scientific
        # computations stack
        install_requires=[
            'musdb>=0.4.0',
            'pandas>=1.0.1',
            'numpy',
            'scipy',
            'simplejson>=3.19.0',
            'soundfile',
            'jsonschema',
            'matplotlib',
            'scikit-learn'
        ],

        package_data={
            'museval-ssir': ['musdb.schema.json'],
        },

        extras_require={  # Optional
            'dev': ['check-manifest'],
            'tests': ['pytest'],
            'docs': [
                'sphinx',
                'sphinx_rtd_theme',
                'recommonmark',
                'numpydoc'
            ],
        },

        tests_require=[
            'pytest',
            'pytest-cov',
            'coverage>=4.4'
        ],

        classifiers=[
            'Development Status :: 4 - Beta',
            'Environment :: Console',
            'Environment :: Plugins',
            'Intended Audience :: Telecommunications Industry',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Topic :: Multimedia :: Sound/Audio :: Analysis',
            'Topic :: Multimedia :: Sound/Audio :: Sound Synthesis'
        ],

        zip_safe=False,
    )
