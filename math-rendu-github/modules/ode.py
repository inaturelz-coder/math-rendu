"""
ode.py — 常微分方程模块
=======================
Module 4 of "Math is Alive" / 《数学是活的》第 4 模块

实现 / Implements:
    - 一阶 ODE：解析 vs Euler vs RK4 / First-order ODE
    - 阻尼振子（二阶）/ Damped harmonic oscillator
    - RLC 电路与质量-弹簧的等价 / RLC vs mass-spring
    - 相图 / Phase portrait
    - SIR 流行病模型 (COVID-style) / SIR epidemic model
    - Euler 与 RK4 收敛阶比较 / Euler vs RK4 order

Li Zhou
2026 年 6 月 / June 2026
MIT License
"""

import numpy as np


# =============================================================================
# 通用 RK4 / Generic RK4 integrator
# =============================================================================
def rk4_step(f, t, y, dt):
    """单步 RK4：y_{n+1} = y_n + (k1+2k2+2k3+k4)·dt/6。
    Single RK4 step.
    """
    k1 = f(t, y)
    k2 = f(t + dt/2, y + dt*k1/2)
    k3 = f(t + dt/2, y + dt*k2/2)
    k4 = f(t + dt, y + dt*k3)
    return y + dt*(k1 + 2*k2 + 2*k3 + k4)/6


def integrate(f, y0, t):
    """RK4 积分 ODE  y' = f(t,y) on grid t.
    Returns array shape (len(t), len(y0)).
    """
    y0 = np.atleast_1d(np.asarray(y0, dtype=float))
    out = np.zeros((len(t), len(y0)))
    out[0] = y0
    for i in range(len(t)-1):
        out[i+1] = rk4_step(f, t[i], out[i], t[i+1]-t[i])
    return out


# =============================================================================
# 演示 1：一阶 ODE / First-Order ODE
# =============================================================================
def demo_first_order():
    """y' = -2y, y(0)=1  →  y = e^{-2t}。
    Test exponential decay.
    """
    print("\n[Demo 1] 一阶 ODE  y'=-2y,  y(0)=1  → y=e^{-2t}")
    t = np.linspace(0, 3, 100)
    sol = integrate(lambda t, y: -2*y, 1.0, t)
    err = np.max(np.abs(sol[:, 0] - np.exp(-2*t)))
    print(f"  RK4 最大误差 / max error = {err:.3e}")


# =============================================================================
# 演示 2：阻尼振子 / Damped Oscillator
# =============================================================================
def demo_damped_oscillator():
    """m x'' + c x' + k x = 0。
    Show under-, critical-, over-damped regimes.
    """
    print("\n[Demo 2] 阻尼振子 / Damped oscillator  mx'' + cx' + kx = 0")
    m, k = 1.0, 4.0
    t = np.linspace(0, 10, 400)
    cs = {'欠阻尼/under': 0.5, '临界/critical': 2*np.sqrt(m*k), '过阻尼/over': 5.0}
    sols = {}
    for label, c in cs.items():
        def f(t, y, c=c):
            x, v = y
            return np.array([v, -(c*v + k*x)/m])
        sols[label] = integrate(f, [1.0, 0.0], t)
        print(f"  {label:18s}  c={c:.3f}  x(10)={sols[label][-1,0]:+.4f}")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
        plt.figure(figsize=(8, 4))
        for label, sol in sols.items():
            plt.plot(t, sol[:, 0], label=label)
        plt.title('Ch4: 阻尼振子 / Damped harmonic oscillator')
        plt.xlabel('t'); plt.ylabel('x(t)')
        plt.grid(alpha=0.3); plt.legend()
        plt.tight_layout()
        plt.savefig('ch04_damped.png', dpi=120); plt.close()
        print("  ✓ 已保存 ch04_damped.png")
    except ImportError:
        pass


# =============================================================================
# 演示 3：RLC vs Mass-Spring
# =============================================================================
def demo_rlc_vs_mass_spring():
    """RLC 电路与质量-弹簧的同构性。
    L↔m, R↔c, 1/C↔k, q↔x。两套方程形式完全一样。
    """
    print("\n[Demo 3] RLC vs Mass-Spring  同构性 / Isomorphism")
    # Mass-spring: m x'' + c x' + k x = 0
    # RLC:        L q'' + R q' + (1/C) q = 0
    L, R, C = 1.0, 0.5, 0.25   # → ω₀² = 1/(LC) = 4, 2ζω₀ = R/L = 0.5
    m, c, k = L, R, 1/C
    print(f"  L={L}, R={R}, C={C}  ↔  m={m}, c={c}, k={k:.2f}")
    omega = np.sqrt(k/m); print(f"  自然频率 ω₀ = {omega:.4f} rad/s")
    t = np.linspace(0, 12, 400)
    sol = integrate(lambda t, y: np.array([y[1], -(c*y[1]+k*y[0])/m]),
                    [1.0, 0.0], t)
    print(f"  q(12) = x(12) = {sol[-1, 0]:+.4f}")


# =============================================================================
# 演示 4：相图 / Phase Portrait
# =============================================================================
def demo_phase_portrait():
    """非线性摆 x'' + sin x = 0 的相图。
    Phase portrait of nonlinear pendulum.
    """
    print("\n[Demo 4] 相图 / Phase portrait — nonlinear pendulum")
    def f(t, y):
        return np.array([y[1], -np.sin(y[0])])
    t = np.linspace(0, 20, 600)
    inits = [(0.5, 0), (1.5, 0), (2.5, 0), (0, 1.5), (0, 2.0), (0, 2.5)]
    trajectories = [integrate(f, ic, t) for ic in inits]
    print(f"  画出 {len(inits)} 条不同初值的轨迹 / Plotted "
          f"{len(inits)} trajectories")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
        plt.figure(figsize=(7, 6))
        for sol, ic in zip(trajectories, inits):
            plt.plot(sol[:, 0], sol[:, 1], alpha=0.7,
                     label=f'ic={ic}')
        plt.xlabel('x'); plt.ylabel("x'")
        plt.title('Ch4: 摆的相图 / Pendulum phase portrait')
        plt.grid(alpha=0.3); plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig('ch04_phase.png', dpi=120); plt.close()
        print("  ✓ 已保存 ch04_phase.png")
    except ImportError:
        pass


# =============================================================================
# 演示 5：SIR 模型 / SIR COVID model
# =============================================================================
def demo_sir_covid():
    """SIR：S'=-βSI/N, I'=βSI/N-γI, R'=γI。
    Track epidemic curve.
    """
    print("\n[Demo 5] SIR 流行病模型 / SIR epidemic")
    N = 1_000_000
    beta, gamma = 0.3, 0.1
    R0 = beta/gamma
    print(f"  N={N:,}  β={beta}  γ={gamma}  R₀={R0:.2f}")
    def f(t, y):
        S, I, R = y
        return np.array([-beta*S*I/N, beta*S*I/N - gamma*I, gamma*I])
    t = np.linspace(0, 200, 400)
    sol = integrate(f, [N-1, 1, 0], t)
    peak_i = sol[:, 1].argmax()
    print(f"  峰值感染 / Peak I = {sol[peak_i,1]:,.0f}  at day {t[peak_i]:.1f}")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
        plt.figure(figsize=(8, 4))
        plt.plot(t, sol[:, 0], label='易感 S')
        plt.plot(t, sol[:, 1], label='感染 I', color='red')
        plt.plot(t, sol[:, 2], label='康复 R', color='green')
        plt.xlabel('天 / Day'); plt.ylabel('人数')
        plt.legend(); plt.grid(alpha=0.3)
        plt.title(f'Ch4: SIR (R₀={R0:.2f})')
        plt.tight_layout()
        plt.savefig('ch04_sir.png', dpi=120); plt.close()
        print("  ✓ 已保存 ch04_sir.png")
    except ImportError:
        pass


# =============================================================================
# 演示 6：Euler vs RK4
# =============================================================================
def demo_euler_vs_rk4():
    """对 y'=-2y 收敛阶对比。Euler 1 阶，RK4 4 阶。
    """
    print("\n[Demo 6] Euler vs RK4 收敛阶 / Convergence order")
    f = lambda t, y: -2*y
    print(f"{'dt':>10} | {'Euler err':>14} | {'RK4 err':>14}")
    print("-" * 46)
    for dt in [0.5, 0.25, 0.125, 0.0625, 0.03125]:
        N = int(2/dt); t = np.linspace(0, 2, N+1)
        # Euler
        ye = 1.0
        for i in range(N):
            ye = ye + dt * f(t[i], ye)
        # RK4
        yr = 1.0
        for i in range(N):
            yr = rk4_step(f, t[i], yr, dt)
        true = np.exp(-2*2)
        print(f"{dt:>10.5f} | {abs(ye-true):>14.6e} | {abs(yr-true):>14.6e}")


# =============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("ode.py — 数学是活的 / 第 4 模块演示")
    print("=" * 60)
    demo_first_order()
    demo_damped_oscillator()
    demo_rlc_vs_mass_spring()
    demo_phase_portrait()
    demo_sir_covid()
    demo_euler_vs_rk4()
    print("\n" + "=" * 60)
    print("演示完毕 / Demo complete.")
    print("=" * 60)
