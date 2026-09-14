pub struct LifNeuron{
    pub v_mem: f64,
    pub v_rest: f64,
    pub v_th: f64,
    pub tau_m: f64,
    pub is_spiking: bool,
}

impl Default for LifNeuron {
    fn default() -> Self {
        Self {
            v_mem: 0.0,
            v_rest: 0.0,      // Потенциал покоя
            v_th: 1.0,        // Порог спайка
            tau_m: 10e-3,     // Постоянная времени 10 мс (0.01 с)
            is_spiking: false,
        }
    }
}

impl LifNeuron {
    pub fn new(v_rest: f64, v_th: f64, tau_m: f64) -> Self {
        Self {
            v_mem: v_rest, // В начале потенциал равен потенциалу покоя
            v_rest,
            v_th,
            tau_m,
            is_spiking: false,
        }
    }
    pub fn update(&mut self, i_in: f64, dt: f64) -> bool{
        let dv = ((self.v_rest-self.v_mem)+ i_in)/self.tau_m;
        self.v_mem += dv *dt;
        if self.v_mem >= self.v_th {
            self.v_mem = self.v_rest;
            self.is_spiking = true;
        } else {
            self.is_spiking = false;
        }
        self.is_spiking



    }

    
}