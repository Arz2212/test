use rand::Rng;
use crate::vteams::{VteamsConfig, VteamsMemristor};
pub struct DifferentialSynapse {
    pub m_plus: VteamsMemristor,
    pub m_minus: VteamsMemristor,
    pub v_read: f64, // Читающее напряжение для передачи спайка (В)
}

impl DifferentialSynapse {
    pub fn new(config: &VteamsConfig, initial_w_plus: f64, initial_w_minus: f64, v_read: f64) -> Self {
        Self {
            m_plus: VteamsMemristor::new(config, initial_w_plus),
            m_minus: VteamsMemristor::new(config, initial_w_minus),
            v_read,
        }
    }
    pub fn new_random(config: &VteamsConfig, v_read: f64) -> Self {
        let mut rng = rand::thread_rng();
        let w_plus = rng.gen_range(config.w_on..=config.w_off);
        let w_minus = rng.gen_range(config.w_on..=config.w_off);

        Self::new(config, w_plus, w_minus, v_read)
    }
    pub fn effective_weight(&self) -> f64{
        self.m_plus.weight() - self.m_minus.weight()
    }
    pub fn calculate_current(&self, pre_spike: bool)-> f64{
        let mut f: f64 = 0.0;
        if pre_spike{
            f = self.v_read * (self.m_plus.conductance() - self.m_minus.conductance())
        }
        f
    }
    pub fn step(&mut self, v_plus: f64, v_minus: f64, dt: f64) {
        self.m_plus.step(v_plus, dt);
        self.m_minus.step(v_minus, dt);
    }

}