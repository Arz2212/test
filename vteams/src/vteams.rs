
// Исправленная структура (с фигурными скобками)

#[derive(Clone, Copy)]
pub struct VteamsConfig {
    pub v_on: f64,        // Порог SET (В), например 0.4
    pub v_off: f64,       // Порог RESET (В), например -0.3
    pub k_on: f64,        // Скорость SET (м/с)
    pub k_off: f64,       // Скорость RESET (м/с)
    pub alpha_on: f64,    // Нелинейность SET
    pub alpha_off: f64,   // Нелинейность RESET
    pub w_on: f64,        // Мин. толщина слоя (соответствует R_on, м)
    pub w_off: f64,       // Макс. толщина слоя (соответствует R_off, м)
    pub r_on: f64,        // Минимальное сопротивление (Ом)
    pub r_off: f64,       // Максимальное сопротивление (Ом)
    pub a_on: f64,        // Параметр окна SET
    pub a_off: f64,
}
impl Default for VteamsConfig {
    fn default() -> Self {
        Self {
            v_on: 0.4,
            v_off: -0.3,
            k_on: 1e-3,
            k_off: -1e-3,
            alpha_on: 3.0,
            alpha_off: 3.0,
            w_on: 0.0,
            w_off: 10e-9,  // 10 нм
            r_on: 100.0,   // 100 Ом
            r_off: 100e3,  // 100 кОм
            a_on: 0.2e-9,  // 0.2 нм
            a_off: 0.2e-9, // 0.2 нм
        }
    }
}

pub struct VteamsMemristor{
    pub config: VteamsConfig,
    w: f64,
}
impl VteamsMemristor{
    
    pub fn new(config: &VteamsConfig, initial_w: f64) -> Self {
        let clamped_w = initial_w.clamp(config.w_on, config.w_off);
        Self {
            config: *config,
            w: clamped_w,
        }
        
    }
    pub fn step(&mut self, v: f64, dt: f64){
        let mut dw :f64 = 0.0;

        if v < self.config.v_off{
            dw = self.config.k_off * ((v/self.config.v_off)-1.0).powf(self.config.alpha_off) * self.f_off();
        }
        else if self.config.v_off <= v && v <= self.config.v_on {
            dw = 0.0;
        }
        else {
            dw = self.config.k_on * ((v/self.config.v_on)-1.0).powf(self.config.alpha_on) * self.f_on();
        }
        self.w = (self.w + dw * dt).clamp(self.config.w_on, self.config.w_off);

    }
    pub fn resistance(&self)-> f64{
        self.config.r_on + ((self.config.r_off - self.config.r_on)/(self.config.w_off - self.config.w_on)) * (self.w - self.config.w_on)


    }
    pub fn conductance(&self) -> f64{
        1.0/(self.resistance())
    }


    fn f_on(&self) -> f64 {
        let exponent = (self.w - self.config.w_on) / self.config.a_on;
        (-exponent.exp()).exp()
    }

    // f_off(w) = exp( -exp( (w_off - w) / a_off ) )
    fn f_off(&self) -> f64 {
        let exponent = (self.config.w_off - self.w) / self.config.a_off;
        (-exponent.exp()).exp()
    }
    pub fn weight(&self) -> f64 {
        let g_min = 1.0 / self.config.r_off;
        let g_max = 1.0 / self.config.r_on;
    
        (self.conductance() - g_min) / (g_max - g_min)
}

}
