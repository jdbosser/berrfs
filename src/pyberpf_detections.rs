use std::{marker::PhantomData, sync::{Arc, Mutex}};

use itertools::Itertools;
use nalgebra::{DMatrix, DVector};
use pyo3::{pyclass, pymethods, types::{PyAnyMethods, PyTypeMethods}, Bound, Py, PyResult, PyRefMut, Python};
use rand::{distributions::Uniform, rngs::{StdRng, ThreadRng}, Rng, SeedableRng};

use numpy::{ndarray::{Array1, Array2, AssignElem}, IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods, ToPyArray};
use statrs::distribution::{Continuous, MultivariateNormal, Normal}; 

use crate::berpf_detections::{BerPFDetections, BirthModel, ClutterLnPDF, LogLikelihood, Model, Motion, State };

// A static wrapper to make the BerPFDetections work in python. 
// No generics allowed. Thus, the FnMut, Fn things are problematic. 
// These are wrapped in boxes and are placed on the heap.

// type Motion         = Box<dyn Fn(&State, &mut StdRng) -> State + Send + Sync>;
type Measurement    = DVector<f64>; 
// type LogLikelihood  = Box<dyn  Fn(&Measurement, &State)->f64 + Send + Sync>;
// type ClutterLnPDF   = Box<dyn Fn(&Measurement)->f64 + Send + Sync>;
// type BirthModel     = Box<dyn Fn(& [Measurement], usize, &mut StdRng) -> Vec<State> + Send + Sync>;

#[derive(Clone, Debug)]
struct ClutterLnPDFS; 
impl<M> ClutterLnPDF<M> for ClutterLnPDFS {
    fn clutter_lnpdf(&self, measurement: &M) -> f64 {
        0.
    }
}

#[derive(Clone, Debug)]
struct MotionS {
    f: DMatrix<f64>,
    g: DMatrix<f64>,
    q: DMatrix<f64>,
}
impl Motion for MotionS {
    fn motion<R: Rng>(&self, state: &State, rng: &mut R) -> State {
        let state_shape = self.q.shape().1;
        

        let mean = (0..state_shape).map(|_| 0.).collect_vec();
        let cov = self.q.data.as_vec().to_vec();
        let process_dist = MultivariateNormal::new(mean, cov).expect("Cannot construct process noise distribution");



        let result = &self.f * state + &self.g * rng.sample(process_dist.clone()); 
        result
    }
}

#[derive(Clone, Debug)]
struct BirthModelS {
    birth_uniform_area: Vec<(f64, f64)>
}
impl BirthModel<DVector<f64>> for BirthModelS {
    fn birth_model<R: Rng>(&self, measurements: &[DVector<f64>], size: usize, rng: &mut R) -> Vec<State> {

        /*
        if measurements.len() < 0 {
            // Split up where to give birth to the particles
            let particles_per_measurement = (measurements.len() as f64 / ( size as f64)).floor() as usize; 
            let rest = size - particles_per_measurement * measurements.len() as usize;

            let first_set = (0..(particles_per_measurement + rest)).map(|_| {
                let m = &measurements[0]; 
                 
                let ex = rng.sample(Normal::new(0., 2.).unwrap()); 
                let ey = rng.sample(Normal::new(0., 2.).unwrap()); 
                let vx = rng.sample(Normal::new(0., 0.1_f64.sqrt()).unwrap()); 
                let vy = rng.sample(Normal::new(0., 0.1_f64.sqrt()).unwrap()); 
                DVector::from_vec(vec![m[0] + ex, vx, m[1] + ey, vy]) 
            }).collect_vec();

            let other_sets = measurements.iter().skip(1).map(|z| {
                (0..particles_per_measurement).map(|_| {
                    let m = z.clone(); 
                    
                    let vx = rng.sample(Normal::new(0., 0.1).unwrap()); 
                    let vy = rng.sample(Normal::new(0., 0.1).unwrap()); 
                    DVector::from_vec(vec![m[0], vx, m[1], vy]) 
                }).collect_vec()
            }).flatten().collect_vec();

            let all_new = first_set.into_iter().chain(other_sets.into_iter()).collect_vec();
            return all_new       
        } 
        else {
            return (0..size).map(|_| {
                let x = rng.sample(Uniform::new(self.birth_uniform_area[0].0, self.birth_uniform_area[0].1));
                let y = rng.sample(Uniform::new(self.birth_uniform_area[2].0, self.birth_uniform_area[2].1));
                let vx = rng.sample(Normal::new(0., 0.1).unwrap()); 
                let vy = rng.sample(Normal::new(0., 0.1).unwrap()); 
                DVector::from_vec(vec![x, vx, y, vy])
            }).collect_vec()
        }
        */
        (0..size).map(|_| {
            DVector::from_vec(self.birth_uniform_area.iter().map(|(l, u)| {
                rng.sample(Uniform::new(l, u))
            }).collect_vec())
        }).collect_vec()
}}

#[derive(Clone, Debug)]
struct LogLikelihoodS {
    h: DMatrix<f64>,
    r: DMatrix<f64>
}
impl LogLikelihood<Measurement> for LogLikelihoodS {
    fn loglikelihood(&self, measurement: &Measurement, state: &State) -> f64 {
        let dist = MultivariateNormal::new((self.h.clone()*state).data.as_vec().to_vec(), self.r.data.as_vec().to_vec()).expect("Error in creating the log likelihood function");
        
        // println!("measurement: {}", measurement);
        // println!("Expected measurement {}", &self.h.clone()*state);
        let r = dist.ln_pdf(measurement);

        // println!("{}", r);
        r
    }
}

#[pyclass(name = "BerPFDetections")]
#[derive(Clone, Debug)]
pub struct PyBerPFDetections {
    rng: StdRng,
    filter: BerPFDetections<MotionS, LogLikelihoodS, DVector<f64>, ClutterLnPDFS, BirthModelS>
}

type PyMat<'a> = PyReadonlyArray2<'a, f64>; 
type PyVec<'a> = PyReadonlyArray1<'a, f64>; 


fn pymat_to_nalgebra(input: PyMat) -> DMatrix<f64> {
    
    let shape = input.shape();
    let (rows, cols) = (shape[0], shape[1]);

    
    let r = DMatrix::from_vec(cols, rows, input.to_vec().expect("Could not extract data from numpy array")).transpose();
    r
}


#[pymethods]
impl PyBerPFDetections {
    #[new]
    fn new<'py>(f: PyMat, g: PyMat, q: PyMat, h: PyMat, r: PyMat, llambda: f64, nsurv: usize, nborn: usize, birth_uniform_area: Vec<(f64, f64)>, pb: f64, ps: f64, pd: f64, initial_prob: Option<f64>, seed: Option<u64>, py: Python<'py>) -> Self {
        
        
        // Create the rng
        let mut rng: StdRng = match seed {
            Some(n) => StdRng::seed_from_u64(n),
            None => StdRng::from_entropy()
        };

        // Parse the matrices to nalgebra
        let f = pymat_to_nalgebra(f);
        let q = pymat_to_nalgebra(q);
        let g = pymat_to_nalgebra(g);
        let h = pymat_to_nalgebra(h);
        let r = pymat_to_nalgebra(r);

        let state_shape = f.shape().1;


        fn constrain<F>(f: F) -> F
        where
            F: for<'a> Fn(&'a State, usize, &'a mut StdRng) -> Vec<State>, 
        {
            f
        }
        
        let birth_model = BirthModelS {
            birth_uniform_area
        };

        let loglikelihood = LogLikelihoodS {
            h, r
        };

        let measurement_type = PhantomData; 

        let rr = Self {
            rng: rng.clone(), 
            filter: BerPFDetections::new(initial_prob, Model{
                birth_model,
                clutter_lnpdf: ClutterLnPDFS {}, 
                lambda: llambda, 
                loglikelihood,  
                measurement_type,  
                motion: MotionS {
                    f, 
                    g, 
                    q
                }, 
                nborn, 
                nsurv, 
                pb, 
                ps,
                pd, 
            })
        };

        let rrf = rr.clone().filter.measurement_update(&[], &mut rng);
        rr
        
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        // This is the equivalent of `self.__class__.__name__` in Python.
        let class_name: String = slf.get_type().qualname()?;
        // To access fields of the Rust struct, we need to borrow the `PyCell`.
        let filter = &slf.borrow().filter; 
        Ok(format!("{}(\n\tprob = {:#?})", class_name, filter.q))
    }

    fn update<'py>(mut slf: PyRefMut<'py, Self>, measurements: Vec<PyVec>) -> PyRefMut<'py, Self> {
        

        // Convert the measurements to dvectors
        let measurements = measurements.iter().map(|m| {
            let m = m.to_vec().expect("could not extract data from numpy");
            DVector::from_vec(m)
        }).collect_vec();

        let mut new_rng = slf.rng.clone(); 
        
        let new_filter = slf.filter.measurement_update(&measurements, &mut new_rng);

        slf.filter = new_filter; 
        slf.rng = new_rng; 
        
        slf

    }

    fn particles<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        
        if self.filter.particles_s.0.len() > 0 {
            let mut dest = Array2::zeros([self.filter.particles_s.0.len(), self.filter.particles_s.0[0].1.len()]); 
            self.filter.particles_s.0.iter().enumerate().for_each(|(n, (w, s))| {
                let mut row = dest.row_mut(n); 
                let rowd = Array1::from_vec(s.data.as_vec().to_vec());
                row.assign(&rowd);
            });
            return dest.into_pyarray_bound(py);
        }
        else {
            let mut dest = Array2::zeros([0, self.filter.model.motion.f.shape().1]); 
            return dest.into_pyarray_bound(py);
        }
    }

    fn weights<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let mut dest: Array1<f64> = Array1::zeros(self.filter.particles_s.0.len()); 
        self.filter.particles_s.0.iter().enumerate().for_each(|(n, (w, s))| {
            dest[n] = *w;
        });

        dest.into_pyarray_bound(py)
    } 

    fn prob(&self) -> f64 {
        self.filter.q
    }

    fn mean<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>>  {
        
        let dmat = self.filter.mean(); 
        let array = dmat.data.as_vec().to_vec(); 

        array.to_pyarray_bound(py)

    }

    fn maxap<'py>(&self, py: Python<'py>)  -> Bound<'py, PyArray1<f64>> {
        let dmat = self.filter.maxap(); 
        let array = dmat.data.as_vec().to_vec(); 

        array.to_pyarray_bound(py)
    }

    fn cov<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {

        let dmat = self.filter.cov(); 
        let array = Array2::from_shape_vec(
            dmat.shape(),
            dmat.data.as_vec().to_vec()
            )
            .unwrap(); 

        array.to_pyarray_bound(py)

    }
    
}
