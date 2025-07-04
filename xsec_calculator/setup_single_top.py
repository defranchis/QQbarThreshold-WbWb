from setuptools import setup, Extension  # type: ignore
from pybind11.setup_helpers import Pybind11Extension, build_ext # type: ignore

ext_modules = [
    Pybind11Extension(
        "xsec_single_top",
        ["checkSingleTop.cpp"],
        include_dirs=["/afs/cern.ch/user/m/mdefranc/work/private/FCC/QQbar_threshold/install/include"],
        libraries=["QQbar_threshold", "gsl", "gslcblas"],
        library_dirs=["/afs/cern.ch/user/m/mdefranc/work/private/FCC/QQbar_threshold/install/lib"],
    ),
]

setup(
    name="xsec_single_top",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)