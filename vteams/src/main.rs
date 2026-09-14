mod vteams;
mod neuron;
mod synapse;
mod network;

use rand::Rng;
use vteams::VteamsConfig;
use network::SnnNetwork;

fn main() {
    println!("=== Инициализация SNN сети (784 -> 1024 -> 128 -> 10) ===");

    // 1. Создаем дефолтный конфиг для мемристоров VTEAMS
    let config = VteamsConfig::default();

    // Напряжение считывания спайка (В)
    let v_read = 0.1;

    // Шаг времени симуляции (с)
    let dt = 1e-3;

    // 2. Создаем сеть
    println!("Генерация слоев и случайно инициализированных синапсов...");
    let mut network = SnnNetwork::new(&config, v_read);
    println!("Сеть успешно создана!");

    // 3. Генерируем тестовый входной вектор (784 спайка с вероятностью 10%)
    let mut rng = rand::thread_rng();
    let input_spikes: Vec<bool> = (0..784)
        .map(|_| rng.gen_bool(0.1))
        .collect();

    let mut active_inputs = input_spikes.iter().filter(|&&s| s).count();
    let mut output_spikes = network.step(&input_spikes, dt);
    for _ in 0..100000{
        active_inputs = input_spikes.iter().filter(|&&s| s).count();
        output_spikes = network.step(&input_spikes, dt);
    }


    // 5. Выводим результаты
    println!("\n=== Результаты работы сети ===");
    println!("Спайки выходного слоя (10 классов): {:?}", output_spikes);
    
    let active_outputs = output_spikes.iter().filter(|&&s| s).count();
    println!("Количество сработавших выходных нейронов: {}", active_outputs);
}