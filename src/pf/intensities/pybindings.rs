use itertools::Itertools;
use nalgebra::DMatrix;
use rand::Rng;
use statrs::distribution::MultivariateNormal;

use crate::pf::{Motion, State};


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


