
pub fn logsumexp(log_numbers: &[f64]) -> f64 {
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
