use pyo3::prelude::*;
use itertools::{self, Itertools}; 

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn bergsf(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    Ok(())
}

fn logsumexp(lognumbers: &[f64]) -> f64 {
    let c = lognumbers.max(); 
    let r: f64 =  lognumbers.iter().map(|v| (v - c).exp()).sum::<f64>().ln(); 

    c + r
}

use statrs::{distribution::MultivariateNormal, statistics::{Distribution, MeanN, Statistics, VarianceN}};
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

trait PDF: Continuous<DVector<f64>, f64> + Distribution<DVector<f64>> {

}

impl<A: Continuous<DVector<f64>, f64> + Distribution<DVector<f64>>> PDF for A {}

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

impl<C> BerGSF<C> {
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
        
        // Return a normalized prediction gmm, e.g. eq. (96)
        GaussianMixture(vec_gmmix).normalize() 
    }
    
    fn measurement_update(mut self, measurements: &[DVector<f64>], p_state: &GaussianMixture) -> Self {
        
        let log_weights = p_state.log_weights(); 
        let weights = log_weights.iter().map(|lw| lw.exp()).collect_vec(); 
        let h = self.models.measurement.h; 
        let r = self.models.measurement.r; 
        
        let qzs: Vec<_> = p_state.0.iter().map(|(w, g)| {

            let eta = (h * g.mean())
                .data.as_vec().to_vec(); // (101)
                                         
            let s = (h * g.variance().unwrap() * h.transpose() + r)
                .data.as_vec().to_vec(); // (102)

            MultivariateNormal::new(eta, s)
                .expect("Measurement model non sym-pos-def matrix")

        }).collect(); 

        let ln_lambda = self.models.lambda.ln(); 

        let c: PDF = self.models.clutter_distribution; 

        let delta_k = self.models.pd * (1. - measurements.iter().map(|z| {
            
            log_weights.iter().zip(qzs.iter())
                .map(|(lw, qz)| {
                    (lw + qz.ln_pdf(z) - ln_lambda - c.ln_pdf(z)).exp()
                }).sum()

        }).sum() ); // (99)
        


        /*|z: DVector<f64>| {
            p_state.0.iter().map(|(w, g)| {
                let eta = h * g.mean(); // (101)
                let s = h * g.variance().unwrap() * h.transpose() + r; 
            }
        };*/

        p_state.0.iter().map(|(w, g)| {
            let eta = h * g.mean(); // (101)
            let s = h * g.variance().unwrap() * h.transpose() + r; 
        
            

        });


        let f = |z: DVector<f64>| {
            
            p_state.0.iter().map(|(w, g)|{
                let eta = h * g.mean(); // (101)
                let s = h * g.variance().unwrap() * h.transpose() + r; 

            } )
                
            


        };

        self.models.pd * (1. - 1.) 
    }

}
