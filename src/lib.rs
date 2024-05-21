use std::error::Error;
use std::marker::PhantomData;

use pyo3::prelude::*;
use pyo3::types::PyList; 
use itertools::{self, Itertools}; 
use numpy::ndarray::{ArrayD, ArrayViewD, ArrayViewMutD, Array1};
use numpy::{IntoPyArray, PyArray, PyArray1, PyArrayDyn, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn bergsf(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_class::<Number>()?; 
    m.add_class::<BerGSFPy>()?; 
    Ok(())
}

fn logsumexp(lognumbers: &[f64]) -> f64 {
    let c = lognumbers.max(); 
    let r: f64 =  lognumbers.iter().map(|v| (v - c).exp()).sum::<f64>().ln(); 

    c + r
}


    // A "tuple" struct
#[pyclass]
struct Number(i32);

#[pymethods]
impl Number {
    #[new]
    fn new(value: i32) -> Self {
        Number(value)
    }
}


type Instance = BerGSF<MultivariateNormal>;
#[pyclass(name = "BerGSF")]
struct BerGSFPy {
    filter: BerGSF<MultivariateNormal> 
}


type PyMat<'a> = PyReadonlyArray2<'a, f64>; 
type PyVec<'a> = PyReadonlyArray1<'a, f64>; 
#[pymethods]
impl BerGSFPy {
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

        
        BerGSFPy {
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


}


use statrs::{distribution::{MultivariateNormal, Uniform}, statistics::{Distribution, MeanN, Statistics, VarianceN}};
type LogWeight = f64;
#[derive(Debug, Clone)]
struct GaussianMixture(Vec<(LogWeight, MultivariateNormal)>);
impl GaussianMixture {
    fn normalize(mut self) -> Self {
        
        let all_weights = self.log_weights();

        let norm = logsumexp(&all_weights);

        for d in self.0.iter_mut() {
            d.0 = d.0 - norm; 
        }

        self
    }

    fn log_weights(&self) -> Vec<f64> {
        self.0.iter().map(|(w, _)| *w).collect()
    }

    fn lnmul(mut self, log_num: f64) -> Self {
         
        for (w, _) in self.0.iter_mut() {
            *w = *w + log_num
        }

        self
    }

    fn evaluate(&self, points: &[DVector<f64>]) -> Vec<f64> {
        
        points.iter().map(|x| {
            self.0.iter().map(|(lnw, g)| {
                g.pdf(x)*(lnw.exp())
            }).sum()
        }).collect()

    }
}

use nalgebra::{DMatrix, DVector};

#[derive(Debug, Clone)]
struct MotionModel {
    f: DMatrix<f64>, 
    q: DMatrix<f64>, 
}
#[derive(Debug, Clone)]
struct MeasurementModel {
    h: DMatrix<f64>, 
    r: DMatrix<f64>,
}

trait PDF: for<'a> Continuous<&'a DVector<f64>, f64> {

}

impl<A: for<'a> Continuous<&'a  DVector<f64>, f64>> PDF for A {}

use statrs::distribution::Continuous; 
#[derive(Debug, Clone)]
struct Model<C: PDF> {
    measurement: MeasurementModel, 
    motion: MotionModel,
    birth_model: GaussianMixture, 
    lambda: f64, // Clutter rate 
    clutter_distribution: C,
    ps: f64, // Survival probability 
    pb: f64, // Birth probability
    pd: f64, // Probability of detection
}

#[derive(Debug, Clone)]
struct BerGSF<C: PDF> {
    models: Model<C>, 
    q: f64, // Current estimate that the bernoulli cardinality is 1
    s: GaussianMixture, // Current estimate on where the component is
}

impl<C: PDF> BerGSF<C> {
    pub fn predict_prob(&self) -> f64 {
        let (ps, pb) = (self.models.ps, self.models.pb); 

        pb * (1. - self.q) + ps * self.q
    }

    pub fn predict_state(&self) -> GaussianMixture {
        
        let (ps, pb) = (self.models.ps, self.models.pb); 
        let q = self.q; 
        let pred_q = self.predict_prob();
        
        // Implements the coefficients in front of the 
        // distributions in (93)
        let mul_birth = pb * (1. - q)/pred_q; 
        let mul_surv = ps * q / pred_q;

        // Create iterators that applies the changed weights
        let gm_birth = self.models.birth_model.0
            .iter()
            .map(|(w, g)| {
                // Implements (92) in log domain
                (w + mul_birth.ln(), g.clone())
            });

        let gm_surv = self.s.0
            .iter()
            .map(|(w, g)| {
                
                // implements (94) and (95)
                let cov = g.variance().unwrap(); 
                let mean = g.mean().unwrap(); 

                let f = &self.models.motion.f; 
                let q = &self.models.motion.q; 

                let p_mean = (f * &mean).data.as_vec().to_vec();
                let p_cov = (q + f * &cov * f.transpose()).data.as_vec().to_vec();

                let p_g = MultivariateNormal::new(p_mean, p_cov)
                    .expect("Covariance not positive definite after prediction");

                (w + mul_surv.ln(), p_g)
            }); 

        let vec_gmmix: Vec<_> = gm_birth.chain(gm_surv).collect(); 
        //let vec_gmmix: Vec<_> = gm_birth.collect();
        
        // Return a normalized prediction gmm, e.g. eq. (96)
        GaussianMixture(vec_gmmix).normalize() 
    }
}
impl<C: PDF>  BerGSF<C>{
    fn measurement_update(&mut self, measurements: &[DVector<f64>]) -> &Self {
        
        let predicted_state = self.predict_state();
        let log_weights = predicted_state.log_weights(); 
        let weights = log_weights.iter().map(|lw| lw.exp()).collect_vec(); 
        let h = &self.models.measurement.h; 
        let r = &self.models.measurement.r; 
        

        type DV = DVector<f64>; 
        type Mat = DMatrix<f64>;
        struct PerGauss {
            eta: DV, 
            s: Mat, 
            q: MultivariateNormal, 
            k: Mat, 
        }
        


        let precalced: Vec<PerGauss> = predicted_state.0.iter().map(|(w, g)| {

            let eta = (h * g.mean().unwrap()); // (101)
            let s = (h * g.variance().unwrap() * h.transpose() + r); // (102)
                                                                     
            let q = MultivariateNormal::new(eta.data.as_vec().to_vec(), (s).data.as_vec().to_vec())
                .expect("Measurement model non sym-pos-def matrix"); 

            let k = (g.variance().unwrap() * h.transpose()) * (&s).clone().try_inverse().expect("Unable to invert the S-matrix."); 

            PerGauss{eta, s, q, k}

        }).collect(); 

        let ln_lambda = self.models.lambda.ln(); 

        let c: &C = &self.models.clutter_distribution; 

        let delta_k: f64 = self.models.pd * (1. - measurements.iter().map(|z| {
            
            log_weights.iter().zip(precalced.iter())
                .map(|(lw, qz)| {
                    (lw + qz.q.ln_pdf(z) - ln_lambda - c.ln_pdf(z)).exp()
                }).sum::<f64>()

        }).sum::<f64>() ); // (99)
        
        let no_det = predicted_state.clone();

        let det: Vec<_> = precalced.iter().zip(predicted_state.0.iter()).cartesian_product(measurements.iter())
            .map(|((pc, pg), z)|{
                
                let pmean = pg.1.mean().unwrap(); 
                let pcov = pg.1.variance().unwrap(); 
                let old_weight = pg.0; 



                let new_mean = pmean + &pc.k * (z - &pc.eta); // (103)
                let new_cov = &pcov - &pc.k * h * pcov.transpose(); // (104)

                let new_weight = old_weight + pc.q.ln_pdf(z) - self.models.lambda.ln() - c.ln_pdf(z); // (98)

                (new_weight, MultivariateNormal::new(new_mean.data.as_vec().to_vec(), new_cov.data.as_vec().to_vec()).expect("Error in creating gaussian"))

            }).collect(); 

        let det = GaussianMixture(det);

        let no_det = no_det.lnmul((1.0 - self.models.pd).ln() - (1.0 - delta_k).ln()); 
        let det = det.lnmul(self.models.pd.ln() - (1.0 - delta_k).ln());

        let new_s = GaussianMixture([no_det.0, det.0].concat());
        
        let pq = (&self).predict_prob();
        let new_q = ((1.0 - delta_k).ln() + pq.ln() - (1.0 - pq * delta_k).ln()).exp();

        self.s = new_s; 
        self.q = new_q; 
        

        self
    }
}
