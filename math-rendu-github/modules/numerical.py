"""
numerical.py — 数值方法
==========================
Module 12 of "Math is Alive" / 《数学是活的》第 12 章

实现 / Implements:
    - Newton 法求平方根
    - 二分法求根
    - 梯形与 Simpson 积分
    - 四阶 Runge-Kutta (RK4)
    - GPS 三维定位
    - 浮点误差案例

数学基础 / Math:
    Newton:   x_{n+1} = x_n - f(x_n)/f'(x_n)
    Simpson:  ∫ ≈ (h/3)(y0 + 4y1 + 2y2 + 4y3 + ... + yN)
    RK4:      4 个斜率加权平均

Li Zhou
2027 / MIT License
"""

import numpy as np


# =============================================================================
# 演示 1：Newton 法开平方
# Demo 1: Newton's method for √a
# =============================================================================
def demo_newton_sqrt():
    """求解 x² = a：x ← (x + a/x) / 2。
    Solve x² = a by Newton iteration.
    """
    print("【演示 1】Newton 法开平方")
    print("[Demo 1] Newton's method for square root")
    print()
    a = 2.0
    x = 1.0
    print(f"  目标: √{a} = {np.sqrt(a):.10f}")
    for i in range(7):
        x = 0.5 * (x + a / x)
        print(f"  迭代 {i+1}: x = {x:.12f},  误差 = "
              f"{abs(x - np.sqrt(a)):.2e}")
    print("  ⇒ 收敛速度 ~ 平方阶 (每步双倍精度位数)")
    print()


# =============================================================================
# 演示 2：二分法求根
# Demo 2: Bisection method
# =============================================================================
def demo_bisection():
    """二分法求 f(x) = 0；只要求 f 连续 + 端点异号。
    Bisection: only needs continuity and a sign change.
    """
    print("【演示 2】二分法求根")
    print("[Demo 2] Bisection for f(x) = 0")
    print()
    f = lambda x: x ** 3 - x - 2  # 真实根 ≈ 1.5214
    a, b = 1.0, 2.0
    assert f(a) * f(b) < 0, "端点必须异号"
    for i in range(20):
        m = 0.5 * (a + b)
        if f(a) * f(m) < 0:
            b = m
        else:
            a = m
    print(f"  方程 x³ - x - 2 = 0")
    print(f"  根 ≈ {0.5 * (a + b):.8f}")
    print(f"  误差 < {(b - a):.2e}")
    print()


# =============================================================================
# 演示 3：梯形 vs Simpson
# Demo 3: Trapezoid vs Simpson
# =============================================================================
def demo_trapezoidal_vs_simpson():
    """∫₀^π sin x dx = 2 —— 比较两种数值积分。
    ∫₀^π sin x dx = 2 — compare two quadratures.
    """
    print("【演示 3】梯形 vs Simpson 积分")
    print("[Demo 3] Trapezoid vs Simpson")
    print()
    true_val = 2.0
    f = np.sin
    for N in [4, 8, 16, 64]:
        x = np.linspace(0, np.pi, N + 1)
        y = f(x)
        h = (np.pi - 0) / N
        trap = h * (y.sum() - 0.5 * (y[0] + y[-1]))
        # Simpson 要 N 偶
        simp = h / 3 * (y[0] + y[-1]
                       + 4 * y[1:-1:2].sum()
                       + 2 * y[2:-1:2].sum())
        print(f"  N = {N:>3d}:  梯形 = {trap:.8f} (err {abs(trap-true_val):.2e})"
              f"   Simpson = {simp:.8f} (err {abs(simp-true_val):.2e})")
    print("  ⇒ Simpson 是 4 阶精度，梯形是 2 阶")
    print()


# =============================================================================
# 演示 4：RK4 求解 ODE
# Demo 4: RK4 for ODE
# =============================================================================
def demo_rk4():
    """解 dy/dt = -y, y(0) = 1，真解 y = e^{-t}。
    Solve dy/dt = -y, y(0) = 1; true y = e^{-t}.
    """
    print("【演示 4】RK4 解 ODE")
    print("[Demo 4] RK4 for ODE")
    print()
    f = lambda t, y: -y
    t0, y0, T, h = 0.0, 1.0, 5.0, 0.1
    n = int(T / h)
    t = np.zeros(n + 1)
    y = np.zeros(n + 1)
    y[0] = y0
    for i in range(n):
        k1 = f(t[i], y[i])
        k2 = f(t[i] + h / 2, y[i] + h / 2 * k1)
        k3 = f(t[i] + h / 2, y[i] + h / 2 * k2)
        k4 = f(t[i] + h, y[i] + h * k3)
        y[i + 1] = y[i] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t[i + 1] = t[i] + h
    err = np.max(np.abs(y - np.exp(-t)))
    print(f"  T = {T}, h = {h}, 步数 = {n}")
    print(f"  y(T) ≈ {y[-1]:.10f}  (真解 {np.exp(-T):.10f})")
    print(f"  最大误差 = {err:.2e}")
    print()
    return t, y


# =============================================================================
# 演示 5：GPS 三维定位
# Demo 5: GPS positioning (3D trilateration)
# =============================================================================
def demo_gps_positioning():
    """从 4 颗卫星的距离测量解出接收机坐标 (x, y, z, bias)。
    Solve receiver (x, y, z, clock bias) from 4 satellite ranges.
    """
    print("【演示 5】GPS 定位 (Gauss-Newton)")
    print("[Demo 5] GPS positioning by Gauss-Newton")
    print()
    rng = np.random.default_rng(7)
    sats = np.array([[15600, 7540, 20140],
                     [18760, 2750, 18610],
                     [17610, 14630, 13480],
                     [19170, 610, 18390]], dtype=float)
    true_pos = np.array([0.0, 0.0, 6370.0])  # 地表 ~6370 km
    true_bias = 0.001  # 时钟偏差（秒等效距离 c·b）
    c = 1.0  # 单位化光速
    ranges = np.linalg.norm(sats - true_pos, axis=1) + c * true_bias
    # 加噪
    ranges += rng.normal(0, 0.01, 4)

    # Gauss-Newton 求解 (x, y, z, b)
    p = np.array([1.0, 1.0, 6000.0, 0.0])
    for _ in range(20):
        d = np.linalg.norm(sats - p[:3], axis=1)
        pred = d + c * p[3]
        # 雅可比
        J = np.zeros((4, 4))
        J[:, :3] = -(sats - p[:3]) / d[:, None]
        J[:, 3] = c
        r = pred - ranges
        dp = np.linalg.lstsq(J, -r, rcond=None)[0]
        p += dp
        if np.linalg.norm(dp) < 1e-9:
            break
    print(f"  真坐标 / true : {true_pos}, 真 bias = {true_bias}")
    print(f"  解算坐标     : {p[:3].round(3)}, bias = {p[3]:.5f}")
    print(f"  定位误差: {np.linalg.norm(p[:3] - true_pos):.4f}")
    print()


# =============================================================================
# 演示 6：浮点误差
# Demo 6: Floating-point errors
# =============================================================================
def demo_floating_point_errors():
    """两个经典坑：0.1 + 0.2 != 0.3，和减法消去。
    Two classic gotchas: 0.1 + 0.2 != 0.3, and catastrophic cancellation.
    """
    print("【演示 6】浮点误差")
    print("[Demo 6] Floating-point pitfalls")
    print()
    print(f"  0.1 + 0.2 = {0.1 + 0.2!r}")
    print(f"  (0.1 + 0.2) == 0.3 ? {0.1 + 0.2 == 0.3}")
    print()
    # 灾难性消去
    # f(x) = (1 - cos x) / x² 在 x → 0 时损失精度
    print("  灾难性消去 / catastrophic cancellation:")
    print("  f(x) = (1 - cos x) / x²,  真值 → 0.5")
    for x in [1e-3, 1e-5, 1e-7, 1e-8]:
        naive = (1 - np.cos(x)) / x ** 2
        # 重写：1 - cos x = 2 sin²(x/2)
        stable = 2 * np.sin(x / 2) ** 2 / x ** 2
        print(f"  x = {x:.0e}: naive = {naive:.6f},  "
              f"stable = {stable:.6f}")
    print("  ⇒ 重写避免相近数相减")
    print()


# =============================================================================
# 主函数 / Main
# =============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("numerical.py — 第 12 章演示 / Chapter 12 Demo")
    print("=" * 60)
    print()
    demo_newton_sqrt()
    demo_bisection()
    demo_trapezoidal_vs_simpson()
    t, y = demo_rk4()
    demo_gps_positioning()
    demo_floating_point_errors()
    print("=" * 60)
    print("演示完毕 / Demo complete.")
    print("=" * 60)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(t, y, 'o-', label='RK4 数值解', markersize=3)
        ax.plot(t, np.exp(-t), 'k--', label='真解 e^{-t}')
        ax.set_xlabel("t")
        ax.set_ylabel("y(t)")
        ax.set_title("Ch12: RK4 解 dy/dt = -y")
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig('ch12_numerical.png', dpi=120, bbox_inches='tight')
        plt.close()
        print("\n已保存可视化: ch12_numerical.png")
    except ImportError:
        print("\n(matplotlib 未安装 — 跳过画图)")
