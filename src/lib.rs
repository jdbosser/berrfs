
use pyo3::prelude::*;

 
pub use gsf::pybindings::gaussian_clutter::*; 
pub use gsf::pybindings::uniform_clutter::*; 

mod utils; 
mod pf; 
mod gsf; 

/// A Python module implemented in Rust.
#[pymodule]
fn berrfs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyBerGSFGaussianClutter>()?; 
    m.add_class::<PyBerGSFUniformClutter>()?; 
    
    use crate::pf::detections::pybindings::PyBerPFDetections; 
    m.add_class::<PyBerPFDetections>()?; 

    Ok(())
}
