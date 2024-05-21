= BerGSF

Stands for _Bernoulli Gaussian Sum Filter_. 

== Development

A development environment may be activated using 
```
nix develop .
```
followed by 
```
maturin develop
```

Entering a python shell should now give you access to the 
python module. An example instance of the BerGSF-struct 
can be constructed using
```python
>>> import bergsf
>>> b = bergsf.example_setup()
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


