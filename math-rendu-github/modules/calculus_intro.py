"""
calculus_intro.py — 微积分初步模块
==================================
Module 1 of "Math is Alive" / 《数学是活的》第 1 模块

实现 / Implements:
    - 极限的数值演示 / Numerical demos of limits
    - 数值微分（前差/中差）/ Numerical derivatives (forward / central)
    - 数值积分（trapz/Simpson）/ Numerical integration (trapz / Simpson)
    - 微积分基本定理 (FTC) 验证 / Verify Fundamental Theorem of Calculus
    - 短信流量峰值识别 / Detect SMS traffic peaks via derivative
    - 自动微分 (autodiff) 对比（可选 torch）/ Compare with PyTorch autodiff

Li Zhou
2026 年 6 月 / June 2026
MIT License
"""

import numpy as np


# =============================================================================
# 演示 1：极限 / Limits
# =============================================================================
def demo_limit():
    """演示 (sin x)/x 在 x → 0 处的极限 = 1。
    Demo: lim_{x→0} sin(x)/x = 1.

    思想 / Idea:
        用越来越小的 h 逼近 0，观察函数值是否收敛。
        Use h → 0 and watch the function value converge.
    """
    print("\n[Demo 1] 极限 / Limit:  lim_{x→0} sin(x)/x")
    hs = [10**(-k) for k in range(1, 8)]
    print(f"{'h':>12} | {'sin(h)/h':>16}")
    print("-" * 34)
    for h in hs:
        print(f"{h:>12.0e} | {np.sin(h)/h:>16.12f}")
    print("→ 收敛到 1 / Converges to 1")

    _try_plot_limit(hs)


def _try_plot_limit(hs):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        xs = np.linspace(-1, 1, 400)
        xs = xs[xs != 0]
        ys = np.sin(xs)/xs
        plt.figure(figsize=(7, 4))
        plt.plot(xs, ys, label='sin(x)/x')
        plt.axhline(1, color='red', ls='--', label='极限 = 1')
        plt.title('Ch1: sin(x)/x 在 0 处的极限 / Limit at 0')
        plt.xlabel('x'); plt.ylabel('sin(x)/x')
        plt.legend(); plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('ch01_limit.png', dpi=120)
        plt.close()
        print("  ✓ 已保存 ch01_limit.png")
    except ImportError:
        pass


# =============================================================================
# 演示 2：数值导数 / Numerical Derivative
# =============================================================================
def demo_derivative():
    """前差与中差对 f(x)=x³ 在 x=2 的导数 (=12) 的逼近。
    Forward & central differences for d(x³)/dx at x=2 (=12).
    """
    print("\n[Demo 2] 数值导数 / Numerical derivative of x³ at x=2 (true=12)")
    f = lambda x: x**3
    x0 = 2.0
    print(f"{'h':>10} | {'forward':>14} | {'central':>14}")
    print("-" * 46)
    for h in [1e-1, 1e-3, 1e-5, 1e-7, 1e-9]:
        fwd = (f(x0+h) - f(x0)) / h
        cen = (f(x0+h) - f(x0-h)) / (2*h)
        print(f"{h:>10.0e} | {fwd:>14.8f} | {cen:>14.8f}")
    print("→ 中差精度更高（2 阶）/ Central is 2nd-order accurate")


# =============================================================================
# 演示 3：数值积分 / Numerical Integration
# =============================================================================
def demo_integral():
    """计算 ∫_0^π sin(x) dx = 2 用 trapz 和 Simpson。
    Compute ∫_0^π sin(x) dx = 2 using trapz and Simpson.
    """
    print("\n[Demo 3] 积分 / Integral  ∫_0^π sin(x) dx  (true = 2)")
    for N in [10, 50, 200, 1000]:
        x = np.linspace(0, np.pi, N)
        y = np.sin(x)
        I_trapz = np.trapz(y, x)
        # Simpson 1/3 rule, requires odd N
        if N % 2 == 0:
            x = np.linspace(0, np.pi, N+1)
            y = np.sin(x)
        h = (x[-1]-x[0])/(len(x)-1)
        I_simp = (h/3)*(y[0]+y[-1]+4*np.sum(y[1:-1:2])+2*np.sum(y[2:-2:2]))
        print(f"  N={N:>5}  trapz={I_trapz:.8f}  simpson={I_simp:.8f}")


# =============================================================================
# 演示 4：微积分基本定理 (FTC) / Fundamental Theorem of Calculus
# =============================================================================
def demo_FTC():
    """验证 d/dx ∫_a^x f(t) dt = f(x)。
    Verify d/dx ∫_a^x f(t) dt = f(x).

    选 f(t) = cos(t)，则 ∫_0^x cos t dt = sin x，d/dx sin x = cos x。
    Pick f(t) = cos t → integral = sin x, derivative gives back cos x.
    """
    print("\n[Demo 4] 微积分基本定理 / FTC  on f(t)=cos t")
    x = np.linspace(0, 2*np.pi, 400)
    f = np.cos(x)
    # cumulative integral via trapz
    F = np.concatenate([[0], np.cumsum((f[1:]+f[:-1])/2 * np.diff(x))])
    # derivative of F using np.gradient
    dF = np.gradient(F, x)
    err = np.max(np.abs(dF - f))
    print(f"  max|d/dx F - f|  = {err:.3e}")
    print("  → FTC 成立 (数值误差极小) / FTC holds within numerical error")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
        plt.figure(figsize=(7, 4))
        plt.plot(x, f, label='f(x)=cos x', lw=2)
        plt.plot(x, F, label='F(x)=∫₀ˣ f', ls='--')
        plt.plot(x, dF, label="F'(x) 数值 ≈ cos x", ls=':')
        plt.legend(); plt.grid(alpha=0.3)
        plt.title('Ch1: 微积分基本定理 / FTC')
        plt.tight_layout()
        plt.savefig('ch01_ftc.png', dpi=120)
        plt.close()
        print("  ✓ 已保存 ch01_ftc.png")
    except ImportError:
        pass


# =============================================================================
# 演示 5：短信流量 / SMS Traffic
# =============================================================================
def demo_sms_traffic():
    """模拟某基站 24h 的短信发送累计量 N(t)，用导数找出峰值时刻。
    Simulate 24h cumulative SMS count, use derivative to find peak hours.
    """
    print("\n[Demo 5] 短信流量峰值 / SMS traffic peak detection")
    t = np.linspace(0, 24, 24*60)  # 每分钟
    # 一个早高峰 + 一个晚高峰，叠加噪声
    rate = (100*np.exp(-((t-9)**2)/1.5) + 150*np.exp(-((t-20)**2)/2)
            + 30 + 5*np.sin(0.5*t))
    N = np.concatenate([[0], np.cumsum(rate[1:]*np.diff(t))])
    dNdt = np.gradient(N, t)

    peak_idx = np.argmax(dNdt)
    print(f"  峰值速率 / Peak rate ≈ {dNdt[peak_idx]:.1f} msg/h "
          f"at t = {t[peak_idx]:.2f} h")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
        a1.plot(t, N); a1.set_ylabel('累计 N(t)'); a1.grid(alpha=0.3)
        a2.plot(t, dNdt, color='orange'); a2.set_ylabel('dN/dt')
        a2.set_xlabel('时间 (h)'); a2.grid(alpha=0.3)
        a2.axvline(t[peak_idx], color='red', ls='--', label=f'峰 ≈ {t[peak_idx]:.1f} h')
        a2.legend()
        plt.suptitle('Ch1: 短信流量 / SMS Traffic')
        plt.tight_layout()
        plt.savefig('ch01_sms.png', dpi=120)
        plt.close()
        print("  ✓ 已保存 ch01_sms.png")
    except ImportError:
        pass


# =============================================================================
# 演示 6：自动微分 / Autodiff (optional torch)
# =============================================================================
def demo_autodiff():
    """对比手算导数、数值导数与自动微分。
    Compare analytical, numerical, and autodiff derivatives.
    """
    print("\n[Demo 6] 自动微分对比 / Autodiff comparison on f(x)=x*sin(x²)")
    f = lambda x: x*np.sin(x**2)
    df_true = lambda x: np.sin(x**2) + 2*x**2*np.cos(x**2)
    x0 = 1.3
    h = 1e-5
    num = (f(x0+h)-f(x0-h))/(2*h)
    ana = df_true(x0)
    print(f"  解析 / analytical = {ana:.8f}")
    print(f"  数值 / numerical  = {num:.8f}")
    try:
        import torch
        xt = torch.tensor(x0, requires_grad=True)
        yt = xt*torch.sin(xt**2)
        yt.backward()
        print(f"  autodiff / torch  = {xt.grad.item():.8f}")
    except ImportError:
        print("  (torch 未安装 — 跳过 autodiff / torch not installed)")


# =============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("calculus_intro.py — 数学是活的 / 第 1 模块演示")
    print("=" * 60)
    demo_limit()
    demo_derivative()
    demo_integral()
    demo_FTC()
    demo_sms_traffic()
    demo_autodiff()
    print("\n" + "=" * 60)
    print("演示完毕 / Demo complete.")
    print("=" * 60)
