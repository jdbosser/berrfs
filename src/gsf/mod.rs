
pub mod pybindings; 

use statrs::{distribution::{MultivariateNormal, Uniform}, statistics::{Distribution, MeanN, Statistics, VarianceN}};
use super::utils::logsumexp; 

use itertools::Itertools; 

type LogWeight = f64;
#[derive(Debug, Clone)]
struct GaussianMixture(Vec<(LogWeight, MultivariateNormal)>);
impl GaussianMixture {
    fn normalize(mut self) -> Self {
        
        let all_weights = self.log_weights();

        let norm = logsumexp(&all_weights);

        for d in self.0.iter_mut() {
            d.0 -= norm; 
        }

        self
    }

    fn log_weights(&self) -> Vec<f64> {
        self.0.iter().map(|(w, _)| *w).collect()
    }

    fn lnmul(mut self, log_num: f64) -> Self {
         
        for (w, _) in self.0.iter_mut() {
            *w += log_num
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

#[derive(Debug, Clone)]
struct MUniform {
    uniforms: Vec<Uniform>
}

impl MUniform {
    fn new(min: DVector<f64>, max: DVector<f64>) -> MUniform {
        let uniforms = min.iter().zip(max.iter()).map(|(mn, ma)|{
            Uniform::new(*mn, *ma)
                .unwrap_or_else(|_| panic!("Cannot create a uniform distribution from {} to {}", mn, ma))
        }).collect();

        MUniform {uniforms}
    }
}

impl<'a> Continuous<&'a DVector<f64>, f64> for MUniform {
    fn pdf(&self, x: &'a DVector<f64>) -> f64 {
        
        x.iter().zip(self.uniforms.iter()).map(|(xx, f)| {
            f.pdf(*xx)
        }).fold(1., |a, x| a * x)

    }

    fn ln_pdf(&self, x: &'a DVector<f64>) -> f64 {
        x.iter().zip(self.uniforms.iter()).map(|(xx, f)| {
            f.ln_pdf(*xx)
        }).sum()
    }
}

fn type_check(cont: impl for<'a> Continuous<&'a DVector<f64>, f64>) {}

fn test() {
    let x: MUniform = todo!();
    let y: MultivariateNormal = todo!();
    type_check(y);
    type_check(x); 
}

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
             //
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
            .filter_map(|(w, g)| {
                
                // implements (94) and (95)
                let cov = g.variance().unwrap(); 
                let mean = g.mean().unwrap(); 

                let f = &self.models.motion.f; 
                let q = &self.models.motion.q; 

                let p_mean = (f * &mean).data.as_vec().to_vec();

                let p_cov = q + f * &cov * f.transpose(); //.data.as_vec().to_vec();
                let p_cov = 1./2. * (p_cov.clone() + p_cov.transpose());


                let p_g = MultivariateNormal::new(p_mean, p_cov.data.as_vec().to_vec())
                    .expect("Covariance not positive definite after prediction");
                if mul_surv == 0.0 {
                    return None
                }
                Some((w + mul_surv.ln(), p_g))
            }); 

        let mut vec_gmmix: Vec<(LogWeight, MultivariateNormal)> = gm_birth.chain(gm_surv).collect(); 
        //let vec_gmmix: Vec<_> = gm_birth.collect();
        
        vec_gmmix.sort_by(|a, b| {
            // This will yield a descending order.
            b.0.total_cmp(&a.0)
        });


        // Return a normalized prediction gmm, e.g. eq. (96)
        GaussianMixture(vec_gmmix).normalize() 
    }
}
impl<C: PDF + std::fmt::Debug>  BerGSF<C>{
    fn measurement_update(&mut self, measurements: &[DVector<f64>]) -> &Self {
        //  println!("ok0");
        let predicted_state = self.predict_state();
        // println!("ok1");

        let log_weights = predicted_state.log_weights(); 
        // println!("ok2");
        let weights = log_weights.iter().map(|lw| lw.exp()).collect_vec(); 
        // println!("ok3");
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

            // println!("ok4");
            let eta = h * g.mean().unwrap(); // (101)
            // println!("ok5");
            let s = h * g.variance().unwrap() * h.transpose() + r; // (102)
                                                                      
            // println!("ok6");
            let q = MultivariateNormal::new(eta.data.as_vec().to_vec(), (s).data.as_vec().to_vec())
                .expect("Measurement model non sym-pos-def matrix"); 

            // println!("ok7");
            let k = (g.variance().unwrap() * h.transpose()) * s.clone().try_inverse().expect("Unable to invert the S-matrix."); 

            // println!("ok8");
            PerGauss{eta, s, q, k}

        }).collect(); 

        let lambda = self.models.lambda; 

        let c: &C = &self.models.clutter_distribution; 

        let delta_k: f64 = self.models.pd * (1. - measurements.iter().map(|z| {
            
            log_weights.iter().zip(precalced.iter())
                .map(|(lw, qz)| {
                    (lw.exp() * qz.q.pdf(z))/(lambda * c.pdf(z))
                }).sum::<f64>()

        }).sum::<f64>() ); // (99)
         
        // println!("ok9");
        let mut no_det = predicted_state.clone().normalize();

        let det: Vec<_> = precalced.iter().zip(predicted_state.0.iter()).cartesian_product(measurements.iter())
            .filter_map(|((pc, pg), z)|{
                
                let pmean = pg.1.mean().unwrap(); 
                let pcov = pg.1.variance().unwrap(); 
                let old_weight = pg.0; 

                if !old_weight.is_finite() {
                    return None
                }

                // println!("ok10");
                let new_mean = pmean + &pc.k * (z - &pc.eta); // (103)
                                                              //
                let new_cov = &pcov - &pc.k * h * pcov.transpose(); // (104)
                
                // Ensure that it is symmetric
                let new_cov = 1. / 2. * (new_cov.clone() + new_cov.transpose());

                // println!("ok11");
                let new_weight = old_weight + pc.q.ln_pdf(z) - self.models.lambda.ln() - c.ln_pdf(z); // (98)
               //  dbg!(&old_weight);
               //  dbg!(&pc.q.ln_pdf(z));
               //  dbg!(&self.models.lambda.ln());
               //  dbg!(&c.ln_pdf(z));

                Some((new_weight, MultivariateNormal::new(new_mean.data.as_vec().to_vec(), new_cov.data.as_vec().to_vec()).expect("Error in creating gaussian")))

            }).collect(); 
        
         
        let det = GaussianMixture(det).normalize();
        
        // Coefficients in front of (98)
        if self.models.pd < 1.0 {
            no_det = no_det.lnmul((1.0 - self.models.pd).ln()); 
        }
        else {
            no_det.0 = vec![]
        }

        // dbg!(&delta_k);
        let det = det.lnmul(self.models.pd.ln());

        let new_s = GaussianMixture([no_det.0, det.0].concat());
        let pq = self.predict_prob();

        // (97)
        let new_q = ((1.0 - delta_k)/(1.0 - pq * delta_k)).min(1.0);

        let mut new_s = new_s.normalize(); 
        new_s.0.sort_by(|a,b| {b.0.total_cmp(&a.0)}); 
        new_s.0.truncate(20); 
        self.s = new_s.normalize(); 
        self.q = new_q; 
        


        self
    }
}

#[cfg(test)]
mod tests {
    
    use super::*; 
    fn create_bergs() -> BerGSF<MUniform> {
        let models = Model {
            measurement: MeasurementModel {
                h: DMatrix::from_vec(2, 4, vec![1., 0., 0., 0., 0., 0., 1.0, 0.]), 
                r: DMatrix::from_vec(2, 2, vec![1., 0., 0., 1.0])
            },
            motion: MotionModel {
                f: DMatrix::from_vec(4, 4, vec![1., 1., 0., 0., 0., 1., 0., 0., 0., 0., 1., 1., 0., 0., 0., 1.]), 
                q: DMatrix::from_vec(4,4, vec![0.0025, 0.005 , 0.    , 0.    , 0.005 , 0.01  , 0.    , 0.    ,
       0.    , 0.    , 0.0025, 0.005 , 0.    , 0.    , 0.005 , 0.01  ])
            }, 
            birth_model: GaussianMixture(vec![(0., MultivariateNormal::new(vec![0., 0., 0., 0.], 
                                                                           vec![
                                                                           100.0, 0., 0., 0., 
                                                                            0., 0.1, 0., 0.,
                                                                            0., 0., 100.0, 0.,
                                                                            0., 0., 0., 0.1]).unwrap() )]),
            lambda: 0.0001, 
            clutter_distribution: MUniform::new(DVector::from_vec(vec![-100., -100.]), DVector::from_vec(vec![100., 100.])), 
            ps: 0.99, 
            pb: 0.1, 
            pd: 1.0,
        };
        let filter = BerGSF{
            models, 
            q: 0.0,
            s: GaussianMixture(vec![])
        };
        filter
    }
    
    #[test]
    fn predict() {
        let filter = create_bergs();

        let pq = filter.predict_prob(); 
        let ps = filter.predict_state(); 

        assert!(ps.0.len() == 1); 
        assert!(ps.0[0].0.is_finite())
    }

    #[test]
    fn update() {
        let mut filter = create_bergs();
        let mut filter2 = filter.clone(); 
        filter.measurement_update(&vec![]);
        
        // assert!(filter.s.0.len() == 1); 
        // assert!(filter.s.0[0].0.is_finite()); 

        filter2.models.pd = 1.0; 
        filter2.measurement_update(&vec![]);
        assert!(filter2.s.0.len() == 0); 
    }

    #[test]
    fn update2() {
        let mut filter = create_bergs();
        filter.measurement_update(&vec![DVector::from_vec(vec![-40.43224192,  24.35469405]), DVector::from_vec(vec![-75.447954  ,  -91.14266698])]);
        filter.measurement_update(&vec![DVector::from_vec(vec![40.43224192,  -24.35469405]), DVector::from_vec(vec![75.447954  ,  91.14266698])]);
    }

}
