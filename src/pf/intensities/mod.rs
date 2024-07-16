use std::marker::PhantomData;

use itertools::Itertools;
use rand::Rng;

use crate::{pf::{predict_particle_weights, sysresample, State}, utils::logsumexp};

use super::{normalize_logweights, predict_particle_positions, predict_prob, set_logweights, BirthModel, Born, LogLikelihood, LogLikelihoodRatio, Motion, Particle, Surviving};

pub mod pybindings;

#[derive(Debug, Clone)]
pub struct Model<Motion, LogLikelihood, Measurement, BirthModel>
{
    pub pb: f64, // probability of birth
    pub ps: f64, // probability of survival
    pub pd: f64, // probability of detection,
    pub motion: Motion, // Motion model
    pub loglikelihood: LogLikelihood, // Measurement model
pub measurement_type: PhantomData<Measurement>, //
    pub nsurv: usize, 
    pub nborn: usize,
    pub birth_model: BirthModel,
     
} 

#[derive(Debug, Clone)]
struct BerPFIntensities<Motion, LogLikelihood, Measurement, BirthModel> {

    pub model: Model<Motion, LogLikelihood, Measurement, BirthModel>, 
    pub q: f64, 
    pub particles_s: Surviving<Vec<Particle>>, 
    pub particles_b: Born<Vec<Particle>>,

}

impl<Mo, Ll, Ms, Bm> BerPFIntensities<Mo, Ll, Ms, Bm> {

    pub fn new(initial_prob: Option<f64>, model: Model<Mo, Ll, Ms, Bm>) -> Self {
        Self {
            model, 
            q: initial_prob.unwrap_or(0.0).min(1.0).max(0.0), 
            particles_s: Surviving(vec![]), 
            particles_b: Born(vec![])
        }
    }

}

fn update_q(predicted_q: f64, ik: f64) -> f64 {

    if predicted_q == 1.0 {
        
        return 1.0

    }
    predicted_q * ik / (1. - predicted_q + predicted_q * ik)
}

fn get_ik<LikFn>(particles: &[Particle], loglik_rat_fn: &LikFn) -> f64
where 
LikFn: Fn(&State) -> f64
{
    let values = particles.iter().map(|(w, p)| {
 
        w + loglik_rat_fn(p)

    }).collect_vec(); 

    logsumexp(&values).exp()
}

fn weight_update<LogLikT>(particles: &[Particle], loglikt: &LogLikT) -> Vec<Particle> 
where LogLikT: Fn(&State) -> f64 {
    
    let new_weights: Vec<f64> = particles.iter().map(|(lnw, s)| {
        lnw + loglikt(s)
    }).collect();  

    let new_weights = normalize_logweights(&new_weights); 
    
    particles.iter().zip(new_weights).map(|((_, s), new_lnw)| {
        (new_lnw, s.clone())
    }).collect()

}

impl<Mo, Ll, Ms, Bm> BerPFIntensities<Mo, Ll, Ms, Bm> where 
Mo: Motion + Clone, 
Ll: LogLikelihood<Ms> + Clone, 
Bm: BirthModel<Ms> + Clone, 
Ms: Clone
{
    pub fn measurement_update<R: Rng>(&self, measurement: &Ms, rng: &mut R, t: &impl LogLikelihood<Ms>) -> Self {

        // Preparations

        // Quickly construct functions that wrap the random number generator
        let mut wrapped_motion = |state: &State| {
            self.model.motion.motion(state, rng)
        };
        
        // Construct a function to evaluate the fitness of the particle. There 
        // is only one measurement, so this is a nice shortcut
        let loglikrat_particle = |state: &State|  {
            self.model.loglikelihood.loglikt(measurement, state) 
            - self.model.loglikelihood.logliknt(measurement)
        }; 


        // Steps from the algorithm


        // Predict existance probability using (28)
        let predicted_q = predict_prob(self.q, self.model.pb, self.model.ps);

        // Draw sample from the motion model
        let predicted_particles_surv = Surviving(
            predict_particle_positions(self.particles_s.as_ref().0, &mut wrapped_motion)
        ); 
        let predicted_particles_born = Born(
            predict_particle_positions(self.particles_b.as_ref().0, &mut wrapped_motion)
        ); 
        
        // Predict the weights
        let predicted_particles: (Surviving<_>, Born<_>) = predict_particle_weights(
            predicted_particles_surv.as_ref(), 
            predicted_particles_born.as_ref(), 
            self.q, self.model.pb, self.model.ps 
        );
        
        // Compute the likelihood that there is no target. 
        let logliknt = self.model.loglikelihood.logliknt(measurement);
        
        // For every particle, calculate the likelihood that 
        // there is a target
        let likts: Vec<_> = predicted_particles.0.iter()
            .chain(predicted_particles.1.iter())
            .map(|(_, particle)| {
                self.model.loglikelihood.loglikt(measurement, particle)
            })
            .collect();  

        // Approximate integral I_k using (83)
        // TODO: Can probably evaluate this without calling clone, for a speedup. 
        let all_particles: Vec<Particle> = [
            predicted_particles.0.0.clone(), 
            predicted_particles.1.0.clone()
        ].concat();
        let ik = get_ik(&all_particles, &loglikrat_particle); 
        
        // Update existance probability
        let updated_q = update_q(predicted_q, ik); 

        // For every particle, update the weights based on the likelihood that there is a target. 
        // Normalize the weights. 
        // Note: The weight_update function also normalizes the weights. 
        let f = |particle: &State| {
            self.model.loglikelihood.loglikt(measurement, particle)
        }; 
        let all_particles = weight_update(&all_particles, &f);

        // Sysresample to take a set of surviving particles
        
        let surviving_particles = sysresample(
            &all_particles, self.model.nsurv, rng
        );

        // Set the weights equal among the survivors. 
        let surviving_particles = set_logweights(
            &surviving_particles, (1.0_f64 / (self.model.nsurv as f64) ).ln()
        );
        
        // Draw a set of birth particles.
        let birth_states = self.model.birth_model.birth_model(
            measurement, self.model.nborn, rng
        );

        // Set the weights equal for all the birth particles. 
        let birth_particles = birth_states.into_iter().map(|state| {
            ((1.0_f64 / (self.model.nborn as f64)).ln(), state)
        }).collect_vec();
        
        // Output a new self. 
        Self {
            model: self.model.clone(), 
            particles_b: Born(birth_particles), 
            particles_s: Surviving(surviving_particles),
            q: updated_q
        }

    }
}

#[cfg(test)]
mod tests {
    
    use statrs::assert_almost_eq;

    use super::*; 

    #[test]
    fn test_update_q() {

        // Creating triplets of inputs, and expected result
        let data = [
            ((0.0, 0.0), 0.0),
            ((1.0, 0.1), 1.0),
            ((1.0, 0.1), 1.0),
            ((1.0, 0.2), 1.0),
            ((1.0, 0.0), 1.0),
            ((0.5, 3.0), (1.5/2.))
        ];

        for d in data.iter() {
            assert_eq!(update_q(d.0.0, d.0.1), d.1)
        }

    }

    #[test]
    fn test_ik() {
        let loglikrat_fn = |state: &State| {
            return state[0] 
        }; 

        let particle_tests: Vec<Particle> = vec![
            (0.5_f64.ln(), State::from_vec(vec![-1.])),
            (0.5_f64.ln(), State::from_vec(vec![-10.])),
        ]; 
        
        let expected = 0.5 * 1./(1.0_f64.exp()) + 0.5 * 1./(10.0_f64.exp()); 

        assert_eq!(expected, get_ik(&particle_tests, &loglikrat_fn));

        let particle_tests: Vec<Particle> = vec![
            (0.5_f64.ln(), State::from_vec(vec![1.])),
            (0.5_f64.ln(), State::from_vec(vec![10.])),
        ]; 
        
        let expected = 0.5 * (1.0_f64.exp()) + 0.5 * (10.0_f64.exp());

        assert_almost_eq!(expected, get_ik(&particle_tests, &loglikrat_fn),10.0_f64.powi(-11));  

        let particle_tests: Vec<Particle> = vec![
        ]; 
        
        let expected = 0.0_f64;

        assert_eq!(expected, get_ik(&particle_tests, &loglikrat_fn));  
    }

    #[test]
    fn test_weight_update() {

        let loglikt_fn = |state: &State| {
            return state[0] 
        }; 

        let particle_tests: Vec<Particle> = vec![
            (0.5_f64.ln(), State::from_vec(vec![-1.])),
            (0.5_f64.ln(), State::from_vec(vec![-5.])),
        ]; 
        
        
        let total = (1.0_f64/5.0_f64.exp() + (-1.0_f64).exp()); 

        let expected: Vec<Particle> = vec![
            (((-1.0_f64).exp() / total).ln(), State::from_vec(vec![-1.])),
            (((-5.0_f64).exp() / total).ln(), State::from_vec(vec![-5.])),
        ]; 


        expected.iter().zip(weight_update(&particle_tests, &loglikt_fn).iter())
            .for_each(|(ep, rp)|{
                assert_almost_eq!(ep.0, rp.0, 10.0_f64.powi(-11))
            })


    }

}
