use std::marker::PhantomData;

use rand::Rng;

use crate::{pf::{predict_particle_weights, State}, utils::logsumexp};

use super::{BirthModel, Born, LogLikelihood, LogLikelihoodRatio, Motion, Particle, Surviving, predict_prob, predict_particle_positions};

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


impl<Mo, Ll, Ms, Bm> BerPFIntensities<Mo, Ll, Ms, Bm> where 
Mo: Motion + Clone, 
Ll: LogLikelihood<Ms> + Clone, 
Bm: BirthModel<Ms> + Clone
{
    pub fn measurement_update<R: Rng>(&self, measurement: &Ms, rng: &mut R, t: &impl LogLikelihood<Ms>) -> Self {

        // Preparations

        // Quickly construct functions that wrap the random number generator
        let mut wrapped_motion = |state: &State| {
            self.model.motion.motion(state, rng)
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
        // TODO: Write a test
        let ik = { // Sprinkling some interior mutability here for speedup.

            let mut maxval = f64::NEG_INFINITY; 
            
            let sum_components: Vec<_> = predicted_particles.0.iter()
                .chain(predicted_particles.1.iter())
                .map(|(logweight, particle)| {

                    let loglikt = self.model.loglikelihood.loglikt(measurement, particle);
                    let loglikrat = loglikt - logliknt; 

                    let val = logweight + loglikrat;
                    val

                }).collect();
            
            logsumexp(&sum_components).exp()
        };
        
        // Update existance probability
        // TODO: Write a test
        let updated_q = predicted_q * ik / (1. - predicted_q + predicted_q * ik); 

        // For every particle, update the weights based on the likelihood that there is a target. 
        
        // Normalize the weights. 

        // Sysresample to take a set of surviving particles
        
        // Set the weights equal among the survivors. 
        
        // Draw a set of birth particles.
        
        // Set the weights equal for all the birth particles. 
        
        // Output a new self. 

        todo!()
    }
}
