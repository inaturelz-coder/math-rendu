"""
probability.py — 概率论
==========================
Module 10 of "Math is Alive" / 《数学是活的》第 10 章

实现 / Implements:
    - 常见分布的采样与可视化
    - 蒙特卡洛估计 π
    - 中心极限定理 (CLT)
    - 马尔可夫链 + 平稳分布
    - 泊松过程
    - 期权定价（Black-Scholes 蒙特卡洛）

数学基础 / Math:
    E[X], Var[X], P(A|B) = P(A∩B)/P(B)
    CLT: 大量 i.i.d. 求和 → 正态
    Black-Scholes:  C = E[max(S_T - K, 0)] e^{-rT}

Li Zhou
2027 / MIT License
"""

import numpy as np


# =============================================================================
# 演示 1：常见分布
# Demo 1: Common distributions
# =============================================================================
def demo_distributions():
    """采样几种基本分布并打印矩。
    Sample several distributions and print their moments.
    """
    print("【演示 1】常见分布")
    print("[Demo 1] Common distributions")
    print()
    rng = np.random.default_rng(0)
    samples = {
        "均匀 / Uniform(0,1)":    rng.uniform(0, 1, 100_000),
        "正态 / Normal(0,1)":     rng.normal(0, 1, 100_000),
        "指数 / Exp(λ=1)":         rng.exponential(1, 100_000),
        "二项 / Binom(n=10,p=0.3)": rng.binomial(10, 0.3, 100_000),
        "泊松 / Poisson(λ=4)":     rng.poisson(4, 100_000),
    }
    for name, s in samples.items():
        print(f"  {name:<32s} 均值 = {s.mean():>6.3f},  "
              f"方差 = {s.var():>6.3f}")
    print()
    return samples


# =============================================================================
# 演示 2：蒙特卡洛估计 π
# Demo 2: Monte Carlo estimation of π
# =============================================================================
def demo_monte_carlo_pi():
    """在单位正方形撒点，落入内接圆的比例 ≈ π/4。
    Throw points in [0,1]², fraction inside quarter circle ≈ π/4.
    """
    print("【演示 2】蒙特卡洛估 π")
    print("[Demo 2] Monte-Carlo estimate of π")
    print()
    rng = np.random.default_rng(1)
    for N in [1_000, 10_000, 100_000, 1_000_000]:
        x = rng.uniform(0, 1, N)
        y = rng.uniform(0, 1, N)
        inside = (x ** 2 + y ** 2 <= 1).sum()
        pi_hat = 4 * inside / N
        err = abs(pi_hat - np.pi)
        print(f"  N = {N:>9d}: π ≈ {pi_hat:.5f}  (误差 {err:.5f})")
    print(f"  ⇒ 误差 ~ 1/√N，是典型 MC 收敛速度")
    print()


# =============================================================================
# 演示 3：中心极限定理 (CLT)
# Demo 3: Central Limit Theorem (CLT)
# =============================================================================
def demo_clt():
    """从非正态分布抽样求和，分布趋于正态。
    Sum samples from a non-normal distribution → goes Gaussian.
    """
    print("【演示 3】中心极限定理")
    print("[Demo 3] Central Limit Theorem")
    print()
    rng = np.random.default_rng(2)
    n_per_sum = 30  # 每次求和的样本数
    n_trials = 50_000
    # 母分布：指数（高度偏斜）
    base = rng.exponential(1, size=(n_trials, n_per_sum))
    means = base.mean(axis=1)
    # 理论：均值 1，方差 = 1/n
    print(f"  母分布 Exp(1): 均值=1, 方差=1")
    print(f"  样本均值（n={n_per_sum}）: "
          f"均值 = {means.mean():.4f}, "
          f"方差 = {means.var():.4f}  "
          f"(理论 1/{n_per_sum} = {1/n_per_sum:.4f})")
    # 与正态比较的偏度
    skew = ((means - means.mean()) ** 3).mean() / means.std() ** 3
    print(f"  偏度 / skewness ≈ {skew:.3f}  (正态为 0)")
    print()
    return means


# =============================================================================
# 演示 4：马尔可夫链 + 平稳分布
# Demo 4: Markov chain + stationary distribution
# =============================================================================
def demo_markov_chain():
    """天气链：晴 / 雨。求稳态。
    Weather chain: sun / rain. Find stationary distribution.
    """
    print("【演示 4】马尔可夫链")
    print("[Demo 4] Markov chain")
    print()
    # 转移矩阵 P[i,j] = P(j | i)
    P = np.array([[0.9, 0.1],   # 晴
                  [0.5, 0.5]])  # 雨
    pi = np.array([1.0, 0.0])  # 起始：晴
    for step in range(1, 31):
        pi = pi @ P
    print(f"  起始: [1, 0]  (今天晴)")
    print(f"  30 步后: {pi.round(4)}")
    # 解 π P = π
    eigvals, eigvecs = np.linalg.eig(P.T)
    stationary = np.real(eigvecs[:, np.argmin(abs(eigvals - 1))])
    stationary /= stationary.sum()
    print(f"  解析平稳分布: {stationary.round(4)}")
    print()


# =============================================================================
# 演示 5：泊松过程
# Demo 5: Poisson process
# =============================================================================
def demo_poisson_process():
    """事件按 λ 强度发生，间隔服从指数分布。
    Events at rate λ, inter-arrival times exponential.
    """
    print("【演示 5】泊松过程")
    print("[Demo 5] Poisson process")
    print()
    rng = np.random.default_rng(3)
    lam = 3.0  # 每秒 3 个事件
    T = 10.0
    intervals = rng.exponential(1 / lam, size=200)
    times = np.cumsum(intervals)
    events = times[times < T]
    print(f"  强度 λ = {lam} 件/秒，时间窗 T = {T}s")
    print(f"  期望事件数 λT = {lam * T:.0f}")
    print(f"  实际事件数 = {len(events)}")
    # 在 [0, T] 上事件数应服从 Poisson(λT)
    print()
    return events


# =============================================================================
# 演示 6：期权定价 (蒙特卡洛 BS)
# Demo 6: Option pricing (Monte-Carlo Black-Scholes)
# =============================================================================
def demo_option_pricing():
    """欧式看涨期权 = E[max(S_T - K, 0)] e^{-rT}。
    European call = discounted expected payoff.
    """
    print("【演示 6】期权定价 (Black-Scholes 蒙特卡洛)")
    print("[Demo 6] Option pricing (Monte-Carlo BS)")
    print()
    S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0
    rng = np.random.default_rng(4)
    Z = rng.standard_normal(200_000)
    S_T = S0 * np.exp((r - 0.5 * sigma ** 2) * T
                      + sigma * np.sqrt(T) * Z)
    payoff = np.maximum(S_T - K, 0)
    price_mc = np.exp(-r * T) * payoff.mean()
    # 解析 Black-Scholes
    from math import erf, sqrt, log, exp
    d1 = (log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    cdf = lambda x: 0.5 * (1 + erf(x / sqrt(2)))
    price_bs = S0 * cdf(d1) - K * exp(-r * T) * cdf(d2)
    print(f"  S0={S0}, K={K}, r={r}, σ={sigma}, T={T}")
    print(f"  蒙特卡洛价格: {price_mc:.4f}")
    print(f"  Black-Scholes 解析: {price_bs:.4f}")
    print(f"  误差: {abs(price_mc - price_bs):.4f}")
    print()


# =============================================================================
# 主函数 / Main
# =============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("probability.py — 第 10 章演示 / Chapter 10 Demo")
    print("=" * 60)
    print()
    samples = demo_distributions()
    demo_monte_carlo_pi()
    means = demo_clt()
    demo_markov_chain()
    events = demo_poisson_process()
    demo_option_pricing()
    print("=" * 60)
    print("演示完毕 / Demo complete.")
    print("=" * 60)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        # 几个分布直方图
        for name, s in list(samples.items())[:3]:
            axes[0].hist(s, bins=40, alpha=0.4, density=True, label=name)
        axes[0].set_title("常见分布")
        axes[0].legend(fontsize=8)

        # CLT
        axes[1].hist(means, bins=60, density=True, alpha=0.7)
        axes[1].set_title("Ch10: CLT —— 样本均值近似正态")

        # 泊松过程事件流
        axes[2].eventplot(events, orientation='horizontal',
                          linewidths=1.5)
        axes[2].set_title("泊松过程事件流")
        axes[2].set_xlabel("t (s)")

        plt.tight_layout()
        plt.savefig('ch10_probability.png', dpi=120, bbox_inches='tight')
        plt.close()
        print("\n已保存可视化: ch10_probability.png")
    except ImportError:
        print("\n(matplotlib 未安装 — 跳过画图)")
