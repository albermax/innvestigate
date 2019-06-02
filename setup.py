import io
import os
import setuptools
import sys


install_requirements = [
    "future",
    "h5py",
    # This package relies on internal interfaces and conventions of Keras.
    # To ensure best compatibility we only support one(, the newest) version.
    "keras==2.2.4",
    "numpy",
    "pillow",
    "pytest",
]

# scipy 1.3 only support py 3.5
if sys.version_info[0] == 2:
    install_requirements += [
        "scipy<1.3"
    ]
else:
    install_requirements += [
        "scipy"
    ]

setup_requirements = [
    "pytest-runner",
]

test_requirements = [
    "pytest"
]


def readme():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    with io.open(
            os.path.join(base_dir, 'README.md'), 'r', encoding='utf-8') as f:
        return f.read()


def setup():
    setuptools.setup(
        name="innvestigate",
        version="1.0.8",
        description="A toolbox to innvestigate neural networks' predictions.",
        long_description=readme(),
        classifiers=[
            "License :: OSI Approved :: BSD License",
            "Development Status :: 5 - Production/Stable",
            "Environment :: Console",
            "Intended Audience :: Science/Research",
            "Operating System :: POSIX :: Linux",
            "Programming Language :: Python :: 2.7",
            "Programming Language :: Python :: 3.6",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
        url="https://github.com/albermax/innvestigate",
        author=("Maxmilian Alber, Sebastian Lapuschkin, Miriam Haegele, " +
                "Kristof Schuett, Philipp Seegerer, Pieter-Jan Kindermans, " +
                "and others"),
        author_email="workDoTalberDoTmaximilian@gmail.com",
        license="BSD-2",
        packages=setuptools.find_packages(),
        install_requires=install_requirements,
        setup_requires=setup_requirements,
        tests_require=test_requirements,
        include_package_data=True,
        zip_safe=False,
    )
    pass


if __name__ == "__main__":
    setup()
