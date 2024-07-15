use itertools::Itertools;
use nalgebra::{DMatrix, DVector};
use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use statrs::distribution::{Continuous, MultivariateNormal};
use pyo3::{prelude::*, types::PyList}; 
use super::super::{BerGSF, GaussianMixture, LogWeight, MeasurementModel, Model, MotionModel};




type Instance = BerGSF<MultivariateNormal>;
#[pyclass(name = "BerGSFGaussianClutter")]
pub struct PyBerGSFGaussianClutter {
    filter: BerGSF<MultivariateNormal> 
}


type PyMat<'a> = PyReadonlyArray2<'a, f64>; 
type PyVec<'a> = PyReadonlyArray1<'a, f64>; 
#[pymethods]
impl PyBerGSFGaussianClutter {
    #[new]
    fn new(f: PyMat, q: PyMat, h: PyMat, r: PyMat, birth_model: &PyList, llambda: f64, clutter_mean: PyVec, clutter_var: PyMat, ps: f64, pb: f64, pd: f64) -> Self {
        
        let dim_x = f.shape()[1];
        let dim_w = q.shape()[1];
        let dim_y = h.shape()[0];
        let dim_e = r.shape()[1];

        let f = f.to_vec().unwrap();
        let q = q.to_vec().unwrap();
        let h = h.to_vec().unwrap();
        let r = r.to_vec().unwrap();

        let gaussians: Vec<(LogWeight, MultivariateNormal)> = birth_model.iter().map(|bm| {
            let mean: PyVec = bm.getattr("mean").unwrap().extract().unwrap();
            let cov: PyMat = bm.getattr("cov").unwrap().extract().unwrap();
            let weight: f64 = bm.getattr("weight").unwrap().extract().unwrap();

            (weight.ln(), MultivariateNormal::new(mean.to_vec().unwrap(), cov.to_vec().unwrap()).unwrap())
        }).collect(); 
        
        let model = Model {
            motion: MotionModel {
                f: DMatrix::from_vec(dim_x, dim_x, f),
                q: DMatrix::from_vec(dim_w, dim_w, q),
            },
            measurement: MeasurementModel {
                h: DMatrix::from_vec(dim_y, dim_x, h),
                r: DMatrix::from_vec(dim_y, dim_y, r),
            },
            lambda: llambda, 
            ps, pb, pd, 
            birth_model: GaussianMixture(gaussians).normalize(), 
            clutter_distribution: MultivariateNormal::new(clutter_mean.to_vec().unwrap(), clutter_var.to_vec().unwrap()).unwrap()
        };

        
        PyBerGSFGaussianClutter {
            filter: BerGSF{
                s: model.birth_model.clone(), 
                models: model, 
                q: 0.
            }
        }
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        // This is the equivalent of `self.__class__.__name__` in Python.
        let class_name: String = slf.get_type().qualname()?;
        // To access fields of the Rust struct, we need to borrow the `PyCell`.
        let filter = &slf.borrow().filter; 
        Ok(format!("{}(\n\tprob = {:#?},\n\t mixtures = {})", class_name, filter.q, filter.s.0.len()))
    }

    fn predict(mut slf: PyRefMut<Self>) -> PyRefMut<Self> {
        slf.filter.q = slf.filter.predict_prob(); 
        slf.filter.s = slf.filter.predict_state(); 
        slf
    }

    fn update<'py>(mut slf: PyRefMut<'py, Self>, measurements: Vec<PyVec>) -> PyRefMut<'py, Self> {
        
        println!("{:#?}", measurements); 

        // Convert the measurements to dvectors
        let measurements = measurements.iter().map(|m| {
            let m = m.to_vec().expect("could not extract data from numpy");
            DVector::from_vec(m)
        }).collect_vec();

        slf.filter.measurement_update(&measurements);
        slf

    }

    fn density<'py>(slf: PyRef<'py, Self>, x: Vec<PyVec<'py>>, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        
        // Evaluate the density at the points of x
        let x: Vec<DVector<f64>> = x.into_iter().map(|xx| {DVector::from_vec(xx.to_vec().expect("could not extract data from numpy"))}).collect();

        let x = slf.filter.s.evaluate(&x);

        x.into_pyarray_bound(py)

    }

    fn density_gaussians<'py>(slf: PyRef<'py, Self>, x: Vec<PyVec<'py>>, py: Python<'py>) -> Vec<Bound<'py, PyArray1<f64>>> {
        
        let data: Vec<_> = slf.filter.s.0.iter().map(|(lnw, g)| {
            
            let v: Vec<f64> = x.iter().map(|xx| {
                g.pdf(&DVector::from_vec(xx.to_vec().expect("cannot extract numpy data"))) * lnw.exp()
            }).collect();
            
           v.into_pyarray_bound(py) 

        }).collect();
        
        data

    }
}
