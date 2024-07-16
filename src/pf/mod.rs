use std::ops::Deref;

use itertools::Itertools;
use nalgebra::{DMatrix, DVector};
use rand::{distributions::Uniform, Rng};

use crate::utils::logsumexp;

pub mod detections; 
pub mod intensities; 

type LogWeight = f64;

pub type State = DVector<f64>;

type Particle = (LogWeight, State);

pub trait Motion {
    fn motion<R: Rng>(&self, state: &State, rng: &mut R) -> State; 
}
pub trait LogLikelihood<M> {
    fn loglikt(&self, measurement: &M, state: &State) -> f64; 
    fn logliknt(&self, measurement: &M) -> f64; 
}

pub trait LogLikelihoodRatio<M> {
    fn loglik_ratio(&self, measurement: &M, state: &State) -> f64; 
}

impl<T: LogLikelihood<M>, M> LogLikelihoodRatio<M> for T {
    fn loglik_ratio(&self, measurement: &M, state: &State) -> f64 {
        self.loglikt(measurement, state) - self.logliknt(measurement)
    }
}

pub trait BirthModel<M> {
    fn birth_model<R: Rng>(&self, measurement: &M, size: usize, rng: &mut R) -> Vec<State>; 
}

// Marker structs
#[derive(Debug, Clone)]
pub struct Surviving<T>(pub T);
impl<U: ?Sized, T: Deref<Target = U>> Surviving<T> {
    fn as_ref(&self) -> Surviving<&U> {
        // Lololol for the ref deref ref
        Surviving(self.0.deref())
    }
}
impl<T> Deref for Surviving<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Debug, Clone)]
pub struct Born<T>(pub T);
impl<U: ?Sized, T: Deref<Target = U>> Born<T> {
    fn as_ref(&self) -> Born<&U> {
        Born(self.0.deref())
    }
}
impl<T> Deref for Born<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}


pub fn predict_prob(prob: f64, pb: f64, ps: f64) -> f64 {
    pb * ( 1. - prob) + ps * prob
}


pub fn predict_particle_positions(
    particles: &[Particle], 
    motion: &mut dyn FnMut(&State) -> State // FnMut since it can contain a mutating rng. 
    ) -> Vec<Particle> {
    particles
        .to_owned()
        .iter_mut()
        .map(|(w, state)| (*w, motion(state)))
        .collect()
}


pub fn predict_particle_weights(
    surviving_particles: Surviving<&[Particle]>, 
    born_particles: Born<&[Particle]>, 
    prob: f64, pb: f64, ps: f64) -> (Surviving<Vec<Particle>>, Born<Vec<Particle>>) {

    let predict_prob = predict_prob(prob, pb, ps); 

    let n_s = surviving_particles.0.len() as f64; 
    let n_b = born_particles.0.len() as f64; 
    let coef_s = ps * prob / predict_prob;
    let coef_b = pb * (1.0 - prob) / predict_prob; 

    let coef_s = coef_s.ln(); 
    let coef_b = coef_b.ln(); 

    let new_sp = surviving_particles.0.iter().map(|(lnw, s)| {
        ((lnw + coef_s), s.clone()) 
    }).collect_vec();
    let new_bp = born_particles.0.iter().map(|(lnw, s)| {
        ((coef_b + (1.0 / n_b).ln()), s.clone())
    }).collect_vec();

    (Surviving(new_sp), Born(new_bp))
}



pub fn set_logweights(particles: &[Particle], new_logweight: f64) -> Vec<Particle> {

    particles.iter().map(|(w, s)| {
        (new_logweight, s.clone())
    }).collect()

}


pub fn normalize_logweights(weights: &[LogWeight]) -> Vec<LogWeight> {
    
    let logsum = logsumexp(weights);

    weights.iter().map(|lnw| {lnw - logsum}).collect()

}



pub fn normalize_particle_weights(particles: &[Particle]) -> Vec<Particle> {

    // Collect the weights 
    let weights = particles.iter().map(|p| p.0).collect_vec(); 
    let normalized_weights = normalize_logweights(&weights);

    normalized_weights.iter().zip(particles.iter()).map(|(nw, p)| {
        (*nw, p.1.clone())
    }).collect_vec()

}


pub fn sysresample_deterministic(particles: &[Particle], n: usize, u_tilde: f64) -> Vec<Particle> {
    
    // Clone the particles. 
    let mut particles: Vec<Particle> = particles.to_owned(); 

    // Sort em
    particles.sort_by(|a, b| {
        let va = a.0; 
        let vb = b.0; 
        va.partial_cmp(&vb)
            .unwrap_or_else(|| panic!("Cannot sort the particles, due to weight a: {va} is not comparable to b: {vb}"))
    });
        
    // particles.iter().for_each(|p| {dbg!(&p.0.exp());} );
        
    // Create the cumulative distribution function
    let f = particles.iter().map(|p| p.0.exp()).scan(0.0, |state, x|{
        *state += x; 

        Some(*state)
    }).collect_vec();
    // dbg!(&f);
    
    // This returns the index of uk
    let f_inv = |u: f64| {
        // u goes between 0. and 1. What particle does this correspond to? 
        //println!("=======");
        // dbg!(&u);
        f.partition_point(|ff| (*ff < u))
    };

    let nf: f64 = n as f64;
    if n < particles.len() {
        let uks = (0..n).map(|k| ((k as f64) + u_tilde)/nf  ); 
        
        // Which bin does the uks end up in?
        let indices = uks.map(f_inv);

        // Create a vector of these particles, and return
        return indices.map(|ii| particles[ii].clone()).collect_vec()
    }

    particles
}


pub fn sysresample<R: Rng + ?Sized>(particles: &[Particle], n: usize, rng:  &mut R) -> Vec<Particle>
{
    let u = rng.sample(Uniform::new(0.0, 1.0));
    sysresample_deterministic(particles, n, u)
}



pub fn mean_particle(particles: &[Particle]) -> State {
    let particles = normalize_particle_weights(particles);
    particles.iter().map(|(lnw, s)| {
        lnw.exp() * s
    }).sum::<State>()
}


pub fn maxap_particle(particles: &[Particle]) -> State {

    let particles = normalize_particle_weights(particles);
    particles.iter().fold(&particles[0], |acc, p| {

        if p.0 > acc.0 {
            p
        } else {
            acc
        }

    }).1.clone()

}

pub fn cov_particles(particles: &[Particle]) -> DMatrix<f64> {

    let particles = normalize_particle_weights(particles);
    let mean = mean_particle(&particles);
    particles.iter().map(|(lnw, s)| {
        lnw.exp() * (s - &mean) * ((s - &mean).transpose())
    }).sum::<DMatrix<f64>>()
}



#[cfg(test)]
mod tests {
    use nalgebra::DMatrix;
    use rand::{distributions::Distribution, thread_rng};
    use statrs::{assert_almost_eq, distribution::MultivariateNormal};

    use super::*; 
    #[test]
    fn test_prob_prediction() {

        assert!(predict_prob(0.5, 0.1, 0.99) == 0.545);
        assert!(predict_prob(0., 0.1, 0.99) == 0.1);
        assert!(predict_prob(1., 0.1, 0.99) == 0.99);
    }

    #[test]
    fn test_predict_particle_positions() {

        // Figured out that creation of matrices are transposed comparet do what I thought
        let f = DMatrix::from_vec(2, 2, vec![1.0, 0.1, 0., 1.]).transpose(); 

        let mut motion = |state: &State| {
            &f * state 
        };

        let predicted_positions: Vec<Particle> = predict_particle_positions(&[], &mut motion);
        assert!(predicted_positions.len() == 0);
        let predicted_positions: Vec<Particle> = predict_particle_positions(&[(0.0, State::from_vec(vec![1.0, 0.0]))], &mut motion);
        dbg!(&predicted_positions);
        assert!(predicted_positions == vec![(0.0, State::from_vec(vec![1.0, 0.0]))]); 
        let predicted_positions: Vec<Particle> = predict_particle_positions(&[(0.5_f64.ln(), State::from_vec(vec![1.0, -1.0])), (0.5_f64.ln(), State::from_vec(vec![1.0, 1.0]))], &mut motion);
        assert!(predicted_positions == vec![(0.5_f64.ln(), State::from_vec(vec![0.9, -1.0])), (0.5_f64.ln(), State::from_vec(vec![1.1, 1.0]))]);

        // Create a new motion struct, that adds some noise to the motion
        let dist = MultivariateNormal::new(vec![0.0, 0.0], vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let mut rng = thread_rng();
        let mut motion = |state: &State| {
            &f * state + dist.sample(&mut rng)
        };

        let predicted_positions: Vec<Particle> = predict_particle_positions(&[(0.0, State::from_vec(vec![1.0, 0.0]))], &mut motion);
        println!("{:#?}", &predicted_positions);
        assert_ne!(predicted_positions, vec![(0.0, State::from_vec(vec![1.0, 0.0]))])

    }


    #[test]
    fn test_predict_particle_weights() {
        let ps = Surviving(vec![(0.0, State::from_vec(vec![1.0]))]);
        let pb = Born(vec![(0.8_f64.ln(), State::from_vec(vec![1.0])), (0.2_f64.ln(), State::from_vec(vec![1.0]))]);

        let prob_s = 0.99; 
        let prob_b = 0.1;
        let prob = 0.2; 

        let (new_s, new_b) = predict_particle_weights(ps.as_ref(), pb.as_ref(), prob, prob_b, prob_s);
        
        dbg!(&new_s);
        let expected_s = (((prob_s * prob )/(prob_s * prob + (1. - prob)*prob_b)).ln() );
        dbg!(expected_s);
        assert!(new_s[0].0 == expected_s); 
        
        let pred_prob = (prob_s * prob + (1. - prob)*prob_b);

        // Note, at weight prediction, the previous weight for the 
        // born particles does not matter. Thus, 0.8 is replaced with 0.5, and the 
        // same thing for 0.2
        let expected_b = vec![
            ((prob_b * (1. - prob) / pred_prob) * 0.5).ln(),
            ((prob_b * (1. - prob) / pred_prob) * 0.5).ln()
        ];
        
        dbg!(&expected_b);
        let got = new_b.iter().map(|(w, _)| *w).collect_vec();
        dbg!(&got);
        assert!(got == expected_b);
    }



    #[test]
    fn test_set_logweights() {
        let particles = vec![
            (0.1_f64.ln(), State::from_vec(vec![0.0])), 
            (0.15_f64.ln(), State::from_vec(vec![1.0])),
            (0.2_f64.ln(), State::from_vec(vec![2.0])),
            (0.55_f64.ln(), State::from_vec(vec![3.0])),
        ];
        
        let new_weight = (1.0_f64/4.0_f64).ln();
        let new_particles = set_logweights(&particles, new_weight);

        for (w, _) in new_particles.iter() {
            assert_eq!(w, &new_weight)
        }
    }


    #[test]
    fn test_normalize_weights() {
        assert_eq!(
            normalize_logweights(&[4.0_f64.ln(), 1.0_f64.ln()]),
            vec![0.8_f64.ln(), 0.2_f64.ln()]
        );
    }


    #[test]
    fn test_normalize_particle_weights() {

        let particles = vec![
            (4.0_f64.ln(), State::from_vec(vec![0.0])), 
            (1.0_f64.ln(), State::from_vec(vec![1.0])),
        ];

        assert_eq!(
            normalize_particle_weights(&particles),
            vec![
            (0.8_f64.ln(), State::from_vec(vec![0.0])), 
            (0.2_f64.ln(), State::from_vec(vec![1.0])),
            ]
        );
    }

    #[test]
    fn test_sysresample_deterministic() {
        
        let particles = vec![
            (0.1_f64.ln(), State::from_vec(vec![0.0])), 
            (0.15_f64.ln(), State::from_vec(vec![1.0])),
            (0.2_f64.ln(), State::from_vec(vec![2.0])),
            (0.55_f64.ln(), State::from_vec(vec![3.0])),
        ];

        let particles = normalize_particle_weights(&particles);

        let u = 0.0;
        
        assert_eq!(sysresample_deterministic(&particles, 1, u)[0].1, State::from_vec(vec![0.0]));
        assert_eq!(sysresample_deterministic(&particles, 1, 0.05)[0].1, State::from_vec(vec![0.0]));
        assert_eq!(sysresample_deterministic(&particles, 1, 0.15)[0].1, State::from_vec(vec![1.0]));
        assert_eq!(sysresample_deterministic(&particles, 1, 0.3)[0].1, State::from_vec(vec![2.0]));
        assert_eq!(sysresample_deterministic(&particles, 1, 0.5)[0].1, State::from_vec(vec![3.0]));
        assert_eq!(sysresample_deterministic(&particles, 1, 0.475)[0].1, State::from_vec(vec![3.0]));
        assert_eq!(sysresample_deterministic(&particles, 2, 0.89).iter().map(|p| p.1.clone()).collect_vec(), vec![State::from_vec(vec![2.0]), State::from_vec(vec![3.0])]);
        assert_eq!(sysresample_deterministic(&particles, 0, 0.89).iter().map(|p| p.1.clone()).collect_vec().len(), 0);
        assert_eq!(sysresample_deterministic(&particles, 5, 0.89).iter().map(|p| p.1.clone()).collect_vec(), particles.iter().map(|p| p.1.clone()).collect_vec());

    }


    #[test]
    fn test_cov_particles(){
        let particles = vec![
            (0.25_f64.ln(), State::from_vec(vec![0.0, 0.0])), 
            (0.25_f64.ln(), State::from_vec(vec![0.0, 1.0])),
            (0.25_f64.ln(), State::from_vec(vec![1.0, 2.0])),
            (0.25_f64.ln(), State::from_vec(vec![-1.0, 3.0])),
        ];
        
        assert_eq!(cov_particles(&particles), DMatrix::from_vec(2, 2, vec![0.5, -0.25, -0.25, 1.25]))
        
    }


    #[test]
    fn test_mean_particle() {
        let particles = vec![
            (0.1_f64.ln(), State::from_vec(vec![0.0, 1.0])), 
            (0.15_f64.ln(), State::from_vec(vec![1.0, -1.0])),
            (0.2_f64.ln(), State::from_vec(vec![2.0, 2.2])),
            (0.55_f64.ln(), State::from_vec(vec![3.0, -4.0])),
        ];

        let result = mean_particle(&particles); 
        let expected = State::from_vec(vec![0.15*1.0 + 0.2*2.0 + 0.55*3.0, 0.1 * 1.0 + 0.15 * (-1.) + 0.2 * 2.2 + 0.55 * (-4.)]);

        assert_almost_eq!(expected[0], result[0], 10. * statrs::prec::DEFAULT_F64_ACC);

    }

    #[test]
    fn test_maxap_particle() {
        let particles = vec![
            (0.1_f64.ln(), State::from_vec(vec![0.0])), 
            (0.15_f64.ln(), State::from_vec(vec![1.0])),
            (0.2_f64.ln(), State::from_vec(vec![2.0])),
            (0.55_f64.ln(), State::from_vec(vec![3.0])),
        ];

        let result = maxap_particle(&particles); 
        let expected = State::from_vec(vec![3.0]);

        assert_eq!(expected, result);
    }
}
