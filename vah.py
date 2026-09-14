from math import exp
import matplotlib.pyplot as plt

# 1. Параметры со слайда
v_set = 2.2        # [В] U_set (переход к R_on)
v_reset = -1.9     # [В] U_reset (переход к R_off)
R_off = 7300.0     # [Ом] 7.3 кОм
R_on = 2100.0      # [Ом] 2.1 кОм

alpha_set = 2.0
alpha_reset = 2.0

a_off = 10e-9      # [м] 10 нм
a_on = 0.0         # [м] 0 нм
w_c = 2e-9         # [м] 2 нм
w_on = 0.0
w_off = 10e-9

# Скорости переключения (подправлены под физический смысл SET/RESET)
k_set = -3e-9      # dw < 0 при v > 2.2V (уменьшаем w до w_on -> R_on)
k_reset = 2e-9     # dw > 0 при v < -1.9V (увеличиваем w до w_off -> R_off)

def gran_fan_off(w):
    return exp(-exp((w - a_off) / w_c))

def gran_fan_on(w):
    return exp(-exp((-w + a_on) / w_c))

def mem_fanc(v, w, dt):
    if v > v_set:
        # Режим SET: dw < 0, ограничение окном gran_fan_on
        dw = k_set * ((v / v_set) - 1.0)**alpha_set * gran_fan_on(w) * dt
    elif v < v_reset:
        # Режим RESET: dw > 0, ограничение окном gran_fan_off
        dw = k_reset * ((v / v_reset) - 1.0)**alpha_reset * gran_fan_off(w) * dt
    else:
        dw = 0.0

    w += dw 

    # ИСПРАВЛЕНИЕ 2: Жесткая отсечка физических границ w
    if w > w_off:
        w = w_off
    elif w < w_on:
        w = w_on

    return w


def main():
    w = w_off      # Начинаем с исходного состояния R_off (7.3 кОм)
    T = 200000       # 2 полных цикла по 1000 шагов
    v = 0.0
    v_l = []
    i_l = []

    # Амплитуда 3.0 В для уверенного пробоя обоих порогов
    V_max = 3.0
    steps_per_cycle = 100000
    dv = (4 * V_max) / steps_per_cycle

    for step in range(T):
        # Формирование пилообразного сигнала [-V_max, +V_max]
        phase = step % steps_per_cycle
        if phase < steps_per_cycle / 4:
            v += dv
        elif phase < 3 * steps_per_cycle / 4:
            v -= dv
        else:
            v += dv

        w = mem_fanc(v, w, dt=1)
        
        # Расчет сопротивления R_M(w)
        x = (w - w_on) / (w_off - w_on)
        R = R_on * ((R_off / R_on) ** x)
        
        i = v / R 
        v_l.append(v)
        i_l.append(i * 1000) # Ток в мА

    # Отрисовка ВАХ
    plt.figure(figsize=(7, 6))
    plt.plot(v_l, i_l, color='b', linestyle='-', linewidth=1.5)
    plt.title("Вольт-амперная характеристика (ВАХ)")
    plt.xlabel("Напряжение V (В)")
    plt.ylabel("Ток I (мА)")
    plt.axhline(0, color='black', lw=0.5, ls='--')
    plt.axvline(0, color='black', lw=0.5, ls='--')
    plt.axvline(v_set, color='green', lw=0.8, ls=':', label='U_set (2.2 V)')
    plt.axvline(v_reset, color='red', lw=0.8, ls=':', label='U_reset (-1.9 V)')
    plt.legend()
    plt.grid(True)
    plt.show()

main()