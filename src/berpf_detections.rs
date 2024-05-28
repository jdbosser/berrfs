use std::ops::Deref;

use itertools::Itertools;
use nalgebra::DVector;
use rand::{distributions::Uniform, Rng};

#[derive(Debug, Clone)]
struct Model<Motion>{
    pb: f64, // probability of birth
    ps: f64, // probability of survival
    motion: Motion         //
} 

type LogWeight = f64;

type State = DVector<f64>;

type Particle = (LogWeight, State);

// Marker structs
#[derive(Debug, Clone)]
struct Surviving<T>(T);
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
struct Born<T>(T);
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

#[derive(Debug, Clone)]
struct BerPFDetections<Motion> {
    model: Model<Motion>,
    q: f64, // Estimated probability
    particles_s: Surviving<Vec<Particle>>, // Surviving particles,
    particles_b: Born<Vec<Particle>>, // Newborn particles
}

fn predict_prob(prob: f64, pb: f64, ps: f64) -> f64 {
    pb * ( 1. - prob) + ps * prob
}

fn predict_particle_positions(
    particles: &[Particle], 
    motion: &mut dyn FnMut(&State) -> State
    ) -> Vec<Particle> {
    particles
        .to_owned()
        .iter_mut()
        .map(|(w, state)| (*w, motion(&state)))
        .collect()
}

fn logsumexp(log_numbers: &[f64]) -> f64 {
    let max: f64 = log_numbers
        .iter()
        .filter(|f| f.is_finite())
        .fold(-f64::INFINITY, |a, b| a.max(*b)); 
    

    let result = {
        match max {
            f64::NEG_INFINITY => f64::NEG_INFINITY,
            _ => {
                let sum: f64 = log_numbers.iter().map(|f| {
                    (f - max).exp()
                 }).sum();

                sum.ln() + max
            }
        }
    };

    result
        
}

fn compute_delta_k<M>(i1: f64, lambda: f64, clutter_lnpdf: &dyn Fn(&M) -> f64, i2fun: &dyn Fn(&M) -> f64,  measurements: &[M] ) -> f64 {
    
    let sumpart = measurements.iter().map(|z| {
        i2fun(z) - lambda.ln() - clutter_lnpdf(z)
    }).collect_vec();
    let sumpart = logsumexp(&sumpart).exp();

    i1.exp() - sumpart
}

fn approximate_i1(pd: f64, particles_s: Surviving<&[Particle]>, particles_b: Born<&[Particle]>) -> f64 {
    let all_particles: &[Particle] = &[particles_s.0, particles_b.0].concat();
    let lnpd = {
        match pd {
            0.0 => f64::NEG_INFINITY,
            _ => pd.ln()
        }
    };
    
    logsumexp(&all_particles.iter().map(|(lnw, _)| {
        lnw + lnpd 
    }).collect_vec())
}

/// Computes the sum in equation (87)
fn weight_update_single_particle_part<M>(
    particle: &Particle, 
    measurements: &[M],
    log_likelihood_fn: &dyn Fn(&M, &State) -> f64, 
    lnlambda: f64, 
    clutter_lnpdf: &dyn Fn(&M) -> f64) -> f64 {
    
    // Computes the sum in equation (87)
    let logs = measurements.iter().map(|z| {
        log_likelihood_fn(z, &particle.1) - lnlambda - clutter_lnpdf(z)
    }).collect_vec();

    logsumexp(&logs)
}

fn weight_update_mut<'a, M>(
        particles: &'a mut [&'a mut Particle], 
        measurements: &[M], 
        pd: f64, 
        lambda: f64, 
        log_likelihood_fn: &dyn Fn(&M, &State) -> f64, 
        clutter_lnpdf: &dyn Fn(&M) -> f64, 
    ) -> &'a mut [&'a mut Particle] {
    
    let lnlambda = lambda.ln(); 

    particles
        .iter_mut()
        .for_each(|particle| {
        let new_weight = (1. - pd + pd * weight_update_single_particle_part(
            *particle, 
            measurements, 
            log_likelihood_fn, 
            lnlambda, 
            clutter_lnpdf, 
        ).exp()).ln() + particle.0;

        (*particle).0 = new_weight; 
    });

    particles
}


fn weight_update<'a, M>(
        particles: &[Particle], 
        measurements: &[M], 
        pd: f64, 
        lambda: f64, 
        log_likelihood_fn: &dyn Fn(&M, &State) -> f64, 
        clutter_lnpdf: &dyn Fn(&M) -> f64, 
    ) -> Vec<Particle> {
    
    let lnlambda = lambda.ln(); 

    particles
        .iter()
        .map(|particle| {
        let new_weight = (1. - pd + pd * weight_update_single_particle_part(
            particle, 
            measurements, 
            log_likelihood_fn, 
            lnlambda, 
            clutter_lnpdf, 
        ).exp()).ln() + particle.0;
        let mut particle = particle.clone();
        particle.0 = new_weight; 
        particle
    }).collect()

}

fn approximate_i2<M>(z: &M, pd: f64, particles_s: Surviving<&[Particle]>, particles_b: Born<&[Particle]>, log_likelihood_fn: &dyn Fn(&M, &State) -> f64) -> f64 {
    let all_particles: &[Particle] = &[particles_s.0, particles_b.0].concat();
    let lnpd = pd.ln(); 
    
    logsumexp(&all_particles.iter().map(|(lnw, s)| {
        lnw + lnpd + log_likelihood_fn(z, s)
    }).collect_vec())
}

fn existance_update(delta_k: f64, predict_prob:f64) -> f64 {
    (1. - delta_k)/(1. - predict_prob * delta_k) * predict_prob
}

fn normalize_weights(weights: &[LogWeight]) -> Vec<LogWeight> {
    
    let logsum = logsumexp(weights);

    weights.iter().map(|lnw| {lnw - logsum}).collect()

}

fn normalize_particle_weights(particles: &[Particle]) -> Vec<Particle> {

    // Collect the weights 
    let weights = particles.iter().map(|p| p.0).collect_vec(); 
    let normalized_weights = normalize_weights(&weights);

    normalized_weights.iter().zip(particles.iter()).map(|(nw, p)| {
        (*nw, p.1.clone())
    }).collect_vec()

}

fn sysresample<R: Rng + ?Sized>(particles: &[Particle], n: usize, rng:  &mut R) -> Vec<Particle>
{
    let u = rng.sample(Uniform::new(0.0, 1.0));
    sysresample_deterministic(particles, n, u)
}

fn sysresample_deterministic(particles: &[Particle], n: usize, u_tilde: f64) -> Vec<Particle> {
    
    // Clone the particles. 
    let mut particles: Vec<Particle> = particles.to_owned(); 

    // Sort em
    particles.sort_by(|a, b| {
        let va = a.0; 
        let vb = b.0; 
        va.partial_cmp(&vb)
            .expect(format!("Cannot sort the particles, due to weight a: {va} is not comparable to b: {vb}").as_str())
    });
        
    particles.iter().for_each(|p| {dbg!(&p.0.exp());} );
        
    // Create the cumulative distribution function
    let f = particles.iter().map(|p| p.0.exp()).scan(0.0, |state, x|{
        *state += x; 

        Some(*state)
    }).collect_vec();
    dbg!(&f);
    
    // This returns the index of uk
    let f_inv = |u: f64| {
        // u goes between 0. and 1. What particle does this correspond to? 
        println!("=======");
        dbg!(&u);
        f.partition_point(|ff| (*ff < u))
    };

    let nf: f64 = n as f64;
    if n < particles.len() {
        let uks = (0..n).map(|k| ((k as f64) + u_tilde)/nf  ); 
        
        // Which bin does the uks end up in?
        let indices = uks.map(|u| f_inv(u));

        indices.clone();

        // Create a vector of these particles, and return
        return indices.map(|ii| particles[ii].clone()).collect_vec()
    }

    particles
}

fn set_logweights(particles: &[Particle], new_logweight: f64) -> Vec<Particle> {

    particles.iter().map(|(w, s)| {
        (new_logweight, s.clone())
    }).collect()

}

fn predict_particle_weights(
    surviving_particles: Surviving<&[Particle]>, 
    born_particles: Born<&[Particle]>, 
    prob: f64, pb: f64, ps: f64) -> (Surviving<Vec<Particle>>, Born<Vec<Particle>>) {

    let predict_prob = predict_prob(prob, pb, ps); 

    let n_s = surviving_particles.0.len() as f64; 
    let n_b = born_particles.0.len() as f64; 
    let coef_s = ((ps * prob / predict_prob));
    let coef_b = (pb * (1.0 - prob) / predict_prob); 

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

impl<Motion> BerPFDetections<Motion> {
    fn update(&mut self) -> &Self {

        // Line 2, equation (28)
        let pred_q = predict_prob(self.q, self.model.pb, self.model.ps);; 
        todo!(); 
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::DMatrix;
    use statrs::assert_almost_eq;
    use statrs::distribution::{MultivariateNormal};
    use rand::distributions::Distribution; 
    use rand::thread_rng; 

    use super::*; 
    #[test]
    fn test_prob_prediction() {

        assert!(predict_prob(0.5, 0.1, 0.99) == 0.545);
        assert!(predict_prob(0., 0.1, 0.99) == 0.1);
        assert!(predict_prob(1., 0.1, 0.99) == 0.99);
    }

    #[test]
    fn test_predict_motion() {

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
    fn test_weight_prediction() {
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
    fn test_i1() {
        let ps = Surviving(vec![(0.0, State::from_vec(vec![1.0]))]);
        let pb = Born(vec![(0.5_f64.ln(), State::from_vec(vec![1.0])), (0.5_f64.ln(), State::from_vec(vec![1.0]))]);
        
        let pd = 0.9;
        let result = approximate_i1(0.9, ps.as_ref(), pb.as_ref());
        let expected: f64 = 0.9 * ( 1.0 + 0.5 + 0.5);

        assert_eq!(result, expected.ln());

        let ps = Surviving(vec![(f64::NEG_INFINITY, State::from_vec(vec![1.0]))]);
        let pb = Born(vec![(0.5_f64.ln(), State::from_vec(vec![1.0])), (0.5_f64.ln(), State::from_vec(vec![1.0]))]);
        
        let result = approximate_i1(0.0, ps.as_ref(), pb.as_ref());
        let expected: f64 = 0.0 * ( 0. + 0.5 + 0.5);

        assert_eq!(result, expected.ln())
    }

    #[test]
    fn test_i2() {
        let pb = Born(vec![(0.0, State::from_vec(vec![2.0]))]);
        let ps = Surviving(vec![(0.8_f64.ln(), State::from_vec(vec![-1.0])), (0.2_f64.ln(), State::from_vec(vec![3.0]))]);
        
       
        // Worlds most simple likelihood function for testing
        let log_likelihood_fn: &dyn Fn(&State, &State) -> f64 = & |z, x| {
            let r:DVector<f64> = (z - x);
            println!("{}",r);
            r[0]
        };


        let pd = 0.9;
        let result = approximate_i2(&State::from_vec(vec![0.]),pd, ps.as_ref(), pb.as_ref(), log_likelihood_fn);
        let expected = (pd * ((1. * (-2.0_f64).exp()) + (1.0_f64.exp() * 0.8) + (0.2 * (-3.0_f64).exp())) );
        
        assert_eq!(result, expected.ln());

    }

    #[test]
    fn test_compute_delta_k() {
        let pb = Born(vec![(0.0, State::from_vec(vec![2.0]))]);
        let ps = Surviving(vec![(0.8_f64.ln(), State::from_vec(vec![-1.0])), (0.2_f64.ln(), State::from_vec(vec![3.0]))]);
        
       
        // Worlds most simple likelihood function for testing
        let log_likelihood_fn: &dyn Fn(&State, &State) -> f64 = & |z, x| {
            let r:DVector<f64> = (z - x);
            println!("{}",r);
            r[0]
        };

        let pd = 0.9;
        let i2fun = |z: &State| {
            approximate_i2(z, pd, ps.as_ref(), pb.as_ref(), log_likelihood_fn)
        };

        let i1 = approximate_i1(pd, ps.as_ref(), pb.as_ref());

        let lambda = 3.0; 

        let clutter_lnpdf = |z: &State| {
            z[0]
        };

        let measurements = vec![State::from_vec(vec![1.0]), State::from_vec(vec![3.0])];

        let results = compute_delta_k(i1, lambda, &clutter_lnpdf, &i2fun, &measurements);
        let expected = approximate_i1(pd, ps.as_ref(), pb.as_ref()).exp() - logsumexp(&measurements.iter().map(|z| i2fun(z) - lambda.ln() - clutter_lnpdf(z)).collect_vec()).exp();

        assert_eq!(results, expected)
    }
    
    
    #[test]
    fn test_update_existance() {
        let delta_k = 0.5; 
        let predict_prob = 0.5; 

        let results = existance_update(delta_k, predict_prob); 
        let expected = (1.0 - delta_k) / (1.0 - delta_k * predict_prob) * predict_prob; 
        assert_eq!(results, expected)
    }
    
    #[test]
    fn test_weight_update() {
        
    
        
        let particles = Surviving(vec![(0.8_f64.ln(), State::from_vec(vec![-1.0])), (0.2_f64.ln(), State::from_vec(vec![3.0]))]);

        let measurements = vec![State::from_vec(vec![1.0]), State::from_vec(vec![3.0])];

        // Worlds most simple likelihood function for testing
        let log_likelihood_fn: &dyn Fn(&State, &State) -> f64 = & |z, x| {
            let r:DVector<f64> = (z - x);
            println!("{}",r);
            r[0]
        };

        let clutter_lnpdf = |z: &State| {
            0.01
        };
        let lambda: f64 = 3.0; 

        let pd = 0.9;
        let expected_w0 = (1.0 - pd + pd * ( 
            (log_likelihood_fn(&measurements[0], &particles[0].1) -  lambda.ln() - clutter_lnpdf(&measurements[0]) ).exp() +
            (log_likelihood_fn(&measurements[1], &particles[0].1) -  lambda.ln() - clutter_lnpdf(&measurements[1]) ).exp()
        )) * particles[0].0.exp();

        let expected_w1= (1.0 - pd + pd * ( 
            (log_likelihood_fn(&measurements[0], &particles[1].1) -  lambda.ln() - clutter_lnpdf(&measurements[0]) ).exp() +
            (log_likelihood_fn(&measurements[1], &particles[1].1) -  lambda.ln() - clutter_lnpdf(&measurements[1]) ).exp()
        )) * particles[1].0.exp();
        

        let expected_inner0 =(
            logsumexp(&[(log_likelihood_fn(&measurements[0], &particles[0].1) -  lambda.ln() - clutter_lnpdf(&measurements[0]) ),
            (log_likelihood_fn(&measurements[1], &particles[0].1) -  lambda.ln() - clutter_lnpdf(&measurements[1]) )]).exp()
        );
        
        // Calculated by hand to be 20.456. 
        dbg!(&expected_inner0);
        assert_eq!(weight_update_single_particle_part(&particles[0], &measurements, &log_likelihood_fn, lambda.ln(), &clutter_lnpdf).exp(), expected_inner0);
        
        // Check that the weight update is correct for all of them. 
        // Some floating point inaccurracies. Need to use assert_almost_eq, 
        // which cannot compare vectors. Thus this whole iter zip shebacle. 
        
        // Calculated by hand to be 14.808
        dbg!(expected_w0);
        (vec![expected_w0, expected_w1])
        .iter().zip( 
        (weight_update(
            &particles.as_ref().0, &measurements, pd, 
            lambda, &log_likelihood_fn, 
            &clutter_lnpdf
        ).iter().map(|p| p.0.exp()))).for_each(|(expected, is)|{

            assert_almost_eq!(*expected, is, 10. * statrs::prec::DEFAULT_F64_ACC)
        });
        
    }
    
    #[test]
    fn test_normalize_weights() {
        assert_eq!(
            normalize_weights(&[4.0_f64.ln(), 1.0_f64.ln()]),
            vec![0.8_f64.ln(), 0.2_f64.ln()]
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
    fn test_set_weights() {
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
}
