import io
import os
import setuptools


install_requirements = [
    "future",
    "h5py",
    "matplotlib",
    "numpy",
    "pillow",
    "scipy",
    "tensorflow>=2.3",
]

setup_requirements = [
    "pytest-runner",
]

test_requirements = [
    "pytest",
    "pytest-cov",
    "pytest-xdist",
]


def readme():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    with io.open(
            os.path.join(base_dir, 'README.md'), 'r', encoding='utf-8') as f:
        return f.read()


def setup():
    setuptools.setup(
        name="innvestigate",
        version="2.0.0",
        description="A toolbox to innvestigate neural networks' predictions.",
        long_description=readme(),
        classifiers=[
            "License :: OSI Approved :: BSD License",
            "Development Status :: 5 - Production/Stable",
            "Environment :: Console",
            "Intended Audience :: Science/Research",
            "Operating System :: POSIX :: Linux",
            "Programming Language :: Python :: 3.7",
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
