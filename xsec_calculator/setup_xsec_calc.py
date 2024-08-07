from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "xsec_calc",
        ["ttThresholdScanISR.cpp"],
        include_dirs=["/afs/cern.ch/user/m/mdefranc/work/private/FCC/QQbar_threshold/install/include"],
        libraries=["QQbar_threshold", "gsl", "gslcblas"],
        library_dirs=["/afs/cern.ch/user/m/mdefranc/work/private/FCC/QQbar_threshold/install/lib"],
    ),
]

setup(
    name="xsec_calc",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)