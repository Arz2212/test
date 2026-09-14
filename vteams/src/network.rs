use rayon::prelude::*;
use crate::neuron::LifNeuron;
use crate::synapse::DifferentialSynapse;
use crate::vteams::VteamsConfig;

pub struct SnnNetwork {
    pub layer0: Vec<LifNeuron>,
    pub layer1: Vec<LifNeuron>,
    pub layer2: Vec<LifNeuron>,
    pub layer3: Vec<LifNeuron>,

    pub syn01: Vec<DifferentialSynapse>,
    pub syn12: Vec<DifferentialSynapse>,
    pub syn23: Vec<DifferentialSynapse>,
}

impl SnnNetwork {
    pub fn new(config: &VteamsConfig, v_read: f64) -> Self {
        let n0 = 784;
        let n1 = 1024;
        let n2 = 128;
        let n3 = 10;

        let layer0 = (0..n0).map(|_| LifNeuron::default()).collect();
        let layer1 = (0..n1).map(|_| LifNeuron::default()).collect();
        let layer2 = (0..n2).map(|_| LifNeuron::default()).collect();
        let layer3 = (0..n3).map(|_| LifNeuron::default()).collect();

        // Параллельная генерация синапсов ускорит инициализацию сети
        let syn01 = (0..n0 * n1)
            .into_par_iter()
            .map(|_| DifferentialSynapse::new_random(config, v_read))
            .collect();

        let syn12 = (0..n1 * n2)
            .into_par_iter()
            .map(|_| DifferentialSynapse::new_random(config, v_read))
            .collect();

        let syn23 = (0..n2 * n3)
            .into_par_iter()
            .map(|_| DifferentialSynapse::new_random(config, v_read))
            .collect();

        Self {
            layer0,
            layer1,
            layer2,
            layer3,
            syn01,
            syn12,
            syn23,
        }
    }

    pub fn step(&mut self, input_spikes: &[bool], dt: f64) -> Vec<bool> {
        // 1. Подача входа
        for (i, &spike) in input_spikes.iter().enumerate() {
            if i < self.layer0.len() {
                self.layer0[i].is_spiking = spike;
            }
        }

        // 2. Проход Слой 0 (784) -> Слой 1 (1024) [Параллельно по нейронам слоя 1]
        let n1 = self.layer1.len();
        let layer0 = &self.layer0;
        let syn01 = &self.syn01;

        self.layer1.par_iter_mut().enumerate().for_each(|(j, neuron)| {
            let mut i_in = 0.0;
            for i in 0..layer0.len() {
                let syn_idx = i * n1 + j;
                i_in += syn01[syn_idx].calculate_current(layer0[i].is_spiking);
            }
            neuron.update(i_in, dt);
        });

        // 3. Проход Слой 1 (1024) -> Слой 2 (128) [Параллельно по нейронам слоя 2]
        let n2 = self.layer2.len();
        let layer1 = &self.layer1;
        let syn12 = &self.syn12;

        self.layer2.par_iter_mut().enumerate().for_each(|(j, neuron)| {
            let mut i_in = 0.0;
            for i in 0..layer1.len() {
                let syn_idx = i * n2 + j;
                i_in += syn12[syn_idx].calculate_current(layer1[i].is_spiking);
            }
            neuron.update(i_in, dt);
        });

        // 4. Проход Слой 2 (128) -> Слой 3 (10) [Обычный цикл, так как 10 нейронов распараллеливать неэффективно]
        let n3 = self.layer3.len();
        for j in 0..n3 {
            let mut i_in = 0.0;
            for i in 0..self.layer2.len() {
                let syn_idx = i * n3 + j;
                i_in += self.syn23[syn_idx].calculate_current(self.layer2[i].is_spiking);
            }
            self.layer3[j].update(i_in, dt);
        }

        self.layer3.iter().map(|n| n.is_spiking).collect()
    }
}