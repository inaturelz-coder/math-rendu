"""
statistics.py — 统计推断
==========================
Module 11 of "Math is Alive" / 《数学是活的》第 11 章

实现 / Implements:
    - 描述统计 (均值、方差、分位数)
    - 点估计与置信区间
    - 假设检验 (t-test, χ²)
    - 简单线性回归
    - A/B 测试
    - 单因素方差分析 (ANOVA)
    - 相关 vs 因果

数学基础 / Math:
    CI: x̄ ± t * s/√n
    t-stat: (x̄ - μ₀) / (s/√n)
    R² = 1 - SS_res / SS_tot

Li Zhou
2027 / MIT License
"""

import numpy as np


# =============================================================================
# 工具：描述统计
# Helper: describe
# =============================================================================
def describe(x, name="x"):
    """打印一组数据的基本统计量。
    Print basic statistics of a dataset.
    """
    x = np.asarray(x, dtype=float)
    print(f"  {name}: n={len(x)}, "
          f"均值 = {x.mean():.4f}, "
          f"标准差 = {x.std(ddof=1):.4f}, "
          f"中位数 = {np.median(x):.4f}, "
          f"min={x.min():.3f}, max={x.max():.3f}")


# =============================================================================
# 演示 1：点估计与置信区间
# Demo 1: Point estimate + confidence interval
# =============================================================================
def demo_estimation():
    """正态样本：估计均值，给 95% CI。
    Estimate the mean of a normal sample, with 95% CI.
    """
    print("【演示 1】点估计与 95% 置信区间")
    print("[Demo 1] Point estimate + 95% CI")
    print()
    rng = np.random.default_rng(0)
    true_mu = 5.0
    x = rng.normal(true_mu, 2.0, 40)
    describe(x, "样本")
    xbar = x.mean()
    se = x.std(ddof=1) / np.sqrt(len(x))
    # 用正态近似 1.96 (大样本)
    ci = (xbar - 1.96 * se, xbar + 1.96 * se)
    print(f"  样本均值 = {xbar:.3f}, SE = {se:.3f}")
    print(f"  95% CI ≈ ({ci[0]:.3f}, {ci[1]:.3f})")
    print(f"  真值 μ = {true_mu} {'落在 CI 内 ✓' if ci[0] < true_mu < ci[1] else '不在 CI 内 ✗'}")
    print()


# =============================================================================
# 演示 2：t-检验
# Demo 2: t-test
# =============================================================================
def demo_hypothesis_test():
    """两样本 t-检验：A 组与 B 组的均值有差异吗？
    Two-sample t-test for difference in means.
    """
    print("【演示 2】两样本 t-检验")
    print("[Demo 2] Two-sample t-test")
    print()
    rng = np.random.default_rng(1)
    a = rng.normal(100, 15, 50)
    b = rng.normal(108, 15, 50)
    # 手动 Welch t
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    na, nb = len(a), len(b)
    t = (ma - mb) / np.sqrt(va / na + vb / nb)
    # 自由度 Welch-Satterthwaite
    df = (va / na + vb / nb) ** 2 / (
        (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    try:
        from scipy.stats import t as t_dist
        p = 2 * (1 - t_dist.cdf(abs(t), df))
    except ImportError:
        # 大样本正态近似
        from math import erf, sqrt
        p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
    print(f"  A 均值 = {ma:.2f},  B 均值 = {mb:.2f}")
    print(f"  t = {t:.3f},  df ≈ {df:.1f},  p = {p:.4f}")
    print(f"  ⇒ {'拒绝 H0 (有差异)' if p < 0.05 else '不拒绝 H0'}")
    print()


# =============================================================================
# 演示 3：简单线性回归
# Demo 3: Simple linear regression
# =============================================================================
def demo_linear_regression():
    """y = a + b x + ε，并报告 R²。
    y = a + b x + ε, report R².
    """
    print("【演示 3】简单线性回归 + R²")
    print("[Demo 3] Simple linear regression + R²")
    print()
    rng = np.random.default_rng(2)
    x = np.linspace(0, 10, 60)
    y = 3.0 + 1.5 * x + rng.normal(0, 1.5, x.size)
    A = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    y_hat = A @ coef
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    print(f"  真实: a=3.0, b=1.5")
    print(f"  拟合: a={coef[0]:.3f}, b={coef[1]:.3f}")
    print(f"  R² = {r2:.4f}")
    print()
    return x, y, coef


# =============================================================================
# 演示 4：A/B 测试
# Demo 4: A/B test
# =============================================================================
def demo_ab_test():
    """两个版本的点击率差异——比例检验。
    Click-through rate difference for two versions — proportion test.
    """
    print("【演示 4】A/B 测试")
    print("[Demo 4] A/B test")
    print()
    # 版本 A: 1000 用户, 100 点击
    # 版本 B: 1000 用户, 130 点击
    n_a, x_a = 1000, 100
    n_b, x_b = 1000, 130
    p_a, p_b = x_a / n_a, x_b / n_b
    p_pool = (x_a + x_b) / (n_a + n_b)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z = (p_b - p_a) / se
    from math import erf, sqrt
    p_val = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    print(f"  A: {x_a}/{n_a} = {p_a:.3%}")
    print(f"  B: {x_b}/{n_b} = {p_b:.3%}")
    print(f"  提升 lift = {(p_b - p_a) * 100:.2f} pp")
    print(f"  z = {z:.3f},  p = {p_val:.4f}")
    print(f"  ⇒ {'B 显著优于 A' if p_val < 0.05 else '差异不显著'}")
    print()


# =============================================================================
# 演示 5：ANOVA —— 三组以上均值
# Demo 5: ANOVA — 3+ group means
# =============================================================================
def demo_anova():
    """单因素 ANOVA。
    One-way ANOVA.
    """
    print("【演示 5】单因素 ANOVA")
    print("[Demo 5] One-way ANOVA")
    print()
    rng = np.random.default_rng(3)
    g1 = rng.normal(10, 2, 30)
    g2 = rng.normal(11, 2, 30)
    g3 = rng.normal(13, 2, 30)
    all_data = np.concatenate([g1, g2, g3])
    overall_mean = all_data.mean()
    # SS_between / SS_within
    ss_between = sum(len(g) * (g.mean() - overall_mean) ** 2
                     for g in [g1, g2, g3])
    ss_within = sum(np.sum((g - g.mean()) ** 2)
                    for g in [g1, g2, g3])
    df_b = 3 - 1
    df_w = len(all_data) - 3
    f = (ss_between / df_b) / (ss_within / df_w)
    print(f"  组均值: {g1.mean():.2f}, {g2.mean():.2f}, {g3.mean():.2f}")
    print(f"  F = {f:.3f},  df = ({df_b}, {df_w})")
    print(f"  ⇒ F 显著大于 1，组间差异不可忽视")
    print()


# =============================================================================
# 演示 6：相关 vs 因果
# Demo 6: Correlation vs causation
# =============================================================================
def demo_correlation_vs_causation():
    """通过引入混淆变量，制造虚假相关。
    Manufacture spurious correlation via a confounder.
    """
    print("【演示 6】相关 ≠ 因果")
    print("[Demo 6] Correlation ≠ causation")
    print()
    rng = np.random.default_rng(4)
    Z = rng.normal(0, 1, 500)     # 季节（混淆）
    X = 2 * Z + rng.normal(0, 0.5, 500)  # 冰淇淋销量
    Y = 3 * Z + rng.normal(0, 0.5, 500)  # 溺水人数
    rho_xy = np.corrcoef(X, Y)[0, 1]
    # 控制 Z 后的偏相关
    # 用残差法
    def resid(a, b):
        slope = np.cov(a, b, ddof=1)[0, 1] / np.var(b, ddof=1)
        return a - slope * b
    rx = resid(X, Z)
    ry = resid(Y, Z)
    rho_partial = np.corrcoef(rx, ry)[0, 1]
    print(f"  Corr(X, Y) = {rho_xy:.3f}   (虚假相关)")
    print(f"  控制 Z 后 = {rho_partial:.3f}   (相关消失)")
    print("  ⇒ 冰淇淋不导致溺水；夏天才是因")
    print()


# =============================================================================
# 主函数 / Main
# =============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("statistics.py — 第 11 章演示 / Chapter 11 Demo")
    print("=" * 60)
    print()
    demo_estimation()
    demo_hypothesis_test()
    x, y, coef = demo_linear_regression()
    demo_ab_test()
    demo_anova()
    demo_correlation_vs_causation()
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
        ax.scatter(x, y, alpha=0.6, label='观测')
        ax.plot(x, coef[0] + coef[1] * x, 'r-', lw=2,
                label=f'拟合 y={coef[0]:.2f}+{coef[1]:.2f}x')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Ch11: 简单线性回归 / Linear regression")
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig('ch11_statistics.png', dpi=120, bbox_inches='tight')
        plt.close()
        print("\n已保存可视化: ch11_statistics.png")
    except ImportError:
        print("\n(matplotlib 未安装 — 跳过画图)")
