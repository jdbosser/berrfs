# BerRFS

Stands for _Bernoulli Random Finite Set_. It is a rust implementation
with python bindings of 
[Ristic et.al. tutorial on Bernoulli filters](https://ba-ngu.vo-au.com/vo/RVVF_Bernoulli_TSP13.pdf).

## Development

A development environment may be activated using 
```bash
nix develop .
```
followed by 
```bash
maturin develop
```
This will construct a virtual environment in your current directory at `.venv`. 
Using maturin, development loop is quick. Change some rust or python code, rerun 
`maturin develop` and you should have an incremental debug build of the project. 

Entering a python shell should now give you access to the 
python module. An example instance of a BerRFS-struct (the Gaussian Sum Filter implementation) can be constructed using
```python
# Python 
import berrfs
b = berrfs.example_setup()
```
on which you can perform prediction steps
```python
b.predict()
```
and measurement updates
```python
b.update([np.array([1.]), np.array([2.])])
```
Note that the measurement update performs a prediction step before applying the information from the filters. 


## Release mode

A shell with a python environment with this module compiled in release mode is
constructed through instead by running
```bash
nix develop .#pyrelease
```
Now, there is no need to use maturin. The library should already be installed in your python 
environment. The build is likely to take a few minutes.  
