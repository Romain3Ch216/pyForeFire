# pyForeFire

pyForeFire is a Python library for dealing with C++ library ForeFire.
It provide numerous ForeFire bindings, written with pybind11 python library.

## Building pyForeFire

### Prerequisite

- pybind11 library (can be installed with homebrew on Mac) [`https://pybind11.readthedocs.io/en/stable/index.html`]
- NetCDF and NetCDF-Cxx libraries (can be installed with homebrew on Mac) [`https://www.unidata.ucar.edu/software/netcdf`]
- ForeFire C++ library and Headers [`https://github.com/forefireAPI/firefront`]

#### Build with Python [Preferred Method]

This will create and install pyforefire package inside current Python interpreter.

```bash
FOREFIRE_DIR=path/to/forefire NETCDF_DIR=path/to/netcdf NETCDF_CXX_DIR=path/to/netcdf_cxx python setup.py install
```

#### Build shared library using cmake

This will create pyforefire shared library inside lib folder.

```bash
cmake . -DFOREFIRE_LIB=path/to/libforefireL.dylib -DFOREFIRE_SRC_DIR=path/to/firefront/src -DNETCDF_INCLUDE_DIR=path/to/netcdf/include -DNETCDFCXX_INCLUDE_DIR=path/to/netcdf-cxx/include/

make
```

## Usage

```python
import pyforefire as forefire

ff = forefire.ForeFire()

sizeX = 300
sizeY = 200
myCmd = "FireDomain[sw=(0.,0.,0.);ne=(%f,%f,0.);t=0.]" % (sizeX, sizeY)
ff.execute(myCmd)
...
```

OR

```bash
python test.py
```