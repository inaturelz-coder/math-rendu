"""
optimization.py — 优化方法
============================
Module 13 of "Math is Alive" / 《数学是活的》第 13 章

实现 / Implements:
    - 黄金分割法 (1D)
    - 梯度下降
    - Newton 法 (Hessian)
    - BFGS (拟 Newton)
    - SGD vs full-batch GD
    - Adam 优化器（深度学习常用）
    - 线性规划 (单纯形 / scipy)
    - KKT 条件演示

数学基础 / Math:
    GD:    x_{k+1} = x_k - α ∇f
    Adam:  m_t, v_t 指数平均一阶/二阶矩
    KKT:   ∇L = 0, μ_i g_i = 0, μ_i ≥ 0

Li Zhou
2027 / MIT License
"""

import numpy as np


# =============================================================================
# 演示 1：黄金分割
# Demo 1: Golden-section search
# =============================================================================
def demo_golden_section():
    """1D 无导数极小化。
    1D derivative-free minimization.
    """
    print("【演示 1】黄金分割法")
    print("[Demo 1] Golden-section search")
    print()
    f = lambda x: (x - 2) ** 2 + 1
    phi = (np.sqrt(5) - 1) / 2  # ≈ 0.618
    a, b = -5.0, 5.0
    for _ in range(40):
        c = b - phi * (b - a)
        d = a + phi * (b - a)
        if f(c) < f(d):
            b = d
        else:
            a = c
    print(f"  f(x) = (x-2)² + 1, 极小点 x* = 2")
    print(f"  数值: x* ≈ {0.5*(a+b):.6f}, f* ≈ {f(0.5*(a+b)):.6f}")
    print()


# =============================================================================
# 演示 2：梯度下降
# Demo 2: Gradient descent
# =============================================================================
def demo_gradient_descent():
    """二次型 f(x) = ½ xᵀ A x。
    Minimize a quadratic form.
    """
    print("【演示 2】梯度下降")
    print("[Demo 2] Gradient descent")
    print()
    A = np.array([[3.0, 0.2], [0.2, 1.0]])
    f = lambda x: 0.5 * x @ A @ x
    grad = lambda x: A @ x
    x = np.array([4.0, -3.0])
    alpha = 0.3
    trace = [x.copy()]
    for _ in range(50):
        x = x - alpha * grad(x)
        trace.append(x.copy())
    trace = np.array(trace)
    print(f"  初值: [4, -3]")
    print(f"  最终 x = {x},  f = {f(x):.2e}")
    print(f"  迭代步数 = {len(trace) - 1}")
    print()
    return trace


# =============================================================================
# 演示 3：Newton 法
# Demo 3: Newton's method (uses Hessian)
# =============================================================================
def demo_newton_method():
    """对二次型一步即可收敛。
    For a quadratic, converges in one step.
    """
    print("【演示 3】Newton 法（用 Hessian）")
    print("[Demo 3] Newton's method")
    print()
    A = np.array([[3.0, 0.2], [0.2, 1.0]])
    x = np.array([4.0, -3.0])
    # Newton 步：x ← x - H^{-1} ∇f
    x_new = x - np.linalg.solve(A, A @ x)
    print(f"  初值 x = {x}")
    print(f"  一步 Newton 后: {x_new}  (应为 0)")
    print()


# =============================================================================
# 演示 4：BFGS （拟 Newton）
# Demo 4: BFGS (quasi-Newton)
# =============================================================================
def demo_bfgs():
    """scipy 的 BFGS 求 Rosenbrock。
    BFGS on the Rosenbrock function via scipy.
    """
    print("【演示 4】BFGS on Rosenbrock")
    print("[Demo 4] BFGS on Rosenbrock")
    print()
    try:
        from scipy.optimize import minimize
        rosen = lambda x: (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
        res = minimize(rosen, x0=[-1.2, 1.0], method='BFGS')
        print(f"  最优点 x* ≈ {res.x.round(4)}  (理论 [1, 1])")
        print(f"  迭代次数 = {res.nit}")
        print(f"  f* = {res.fun:.2e}")
    except ImportError:
        print("  (scipy 未安装，跳过)")
    print()


# =============================================================================
# 演示 5：SGD vs GD
# Demo 5: SGD vs full-batch GD
# =============================================================================
def demo_sgd_vs_gd():
    """线性回归损失：SGD 在大数据上更高效但有噪声。
    Linear-regression loss: SGD is noisier but cheaper per step.
    """
    print("【演示 5】SGD vs full-batch GD")
    print("[Demo 5] SGD vs full-batch GD")
    print()
    rng = np.random.default_rng(0)
    n, d = 1000, 5
    X = rng.standard_normal((n, d))
    w_true = rng.standard_normal(d)
    y = X @ w_true + 0.1 * rng.standard_normal(n)

    # full-batch GD
    w = np.zeros(d)
    losses_gd = []
    for _ in range(80):
        g = X.T @ (X @ w - y) / n
        w -= 0.05 * g
        losses_gd.append(np.mean((X @ w - y) ** 2))

    # SGD
    w2 = np.zeros(d)
    losses_sgd = []
    for epoch in range(80):
        idx = rng.permutation(n)
        for i in idx[:64]:  # mini-batch of 1 (true SGD)
            xi = X[i]
            yi = y[i]
            g = (xi @ w2 - yi) * xi
            w2 -= 0.02 * g
        losses_sgd.append(np.mean((X @ w2 - y) ** 2))

    print(f"  GD  最终 loss = {losses_gd[-1]:.4f}")
    print(f"  SGD 最终 loss = {losses_sgd[-1]:.4f}")
    print("  SGD 每步便宜得多，但 loss 曲线波动")
    print()
    return losses_gd, losses_sgd


# =============================================================================
# 演示 6：Adam 优化器
# Demo 6: Adam optimizer
# =============================================================================
def demo_adam():
    """Adam 在 Rosenbrock 上的可视化教程。
    Adam tutorial on the Rosenbrock function.
    """
    print("【演示 6】Adam 优化器")
    print("[Demo 6] Adam optimizer")
    print()
    rosen = lambda x: (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
    grad = lambda x: np.array([
        -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2),
        200 * (x[1] - x[0] ** 2)
    ])
    x = np.array([-1.2, 1.0])
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    alpha, b1, b2, eps = 0.05, 0.9, 0.999, 1e-8
    for t in range(1, 2001):
        g = grad(x)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g ** 2
        m_hat = m / (1 - b1 ** t)
        v_hat = v / (1 - b2 ** t)
        x -= alpha * m_hat / (np.sqrt(v_hat) + eps)
    print(f"  Adam 最优点: {x.round(4)}  (目标 [1, 1])")
    print(f"  f* = {rosen(x):.2e}")
    print()


# =============================================================================
# 演示 7：线性规划
# Demo 7: Linear programming
# =============================================================================
def demo_linear_programming():
    """工厂排产：max 3 x1 + 5 x2 s.t. 约束。
    Production LP via scipy.
    """
    print("【演示 7】线性规划")
    print("[Demo 7] Linear programming")
    print()
    try:
        from scipy.optimize import linprog
        # max 3 x1 + 5 x2  ⇒  min -3 x1 -5 x2
        c = [-3, -5]
        A_ub = [[1, 0], [0, 2], [3, 2]]
        b_ub = [4, 12, 18]
        res = linprog(c, A_ub=A_ub, b_ub=b_ub,
                      bounds=[(0, None), (0, None)])
        print(f"  x* = {res.x.round(3)}")
        print(f"  最大值 = {-res.fun:.3f}")
    except ImportError:
        print("  (scipy 未安装，跳过)")
    print()


# =============================================================================
# 演示 8：KKT 条件
# Demo 8: KKT conditions
# =============================================================================
def demo_kkt_conditions():
    """min x²+y² s.t. x+y ≥ 1。
    A simple inequality-constrained problem; KKT gives the answer.
    """
    print("【演示 8】KKT 条件")
    print("[Demo 8] KKT conditions")
    print()
    # 拉格朗日: L = x² + y² - μ(x + y - 1),  μ ≥ 0
    # ∂L/∂x = 2x - μ = 0  ⇒ x = μ/2
    # ∂L/∂y = 2y - μ = 0  ⇒ y = μ/2
    # 互补松弛: μ(x + y - 1) = 0
    # 若约束紧: x + y = 1 ⇒ μ = 1 ⇒ x = y = 0.5
    x, y, mu = 0.5, 0.5, 1.0
    print(f"  最优 (x*, y*) = ({x}, {y})")
    print(f"  乘子 μ* = {mu}")
    print(f"  验证: 2x - μ = {2*x - mu},  2y - μ = {2*y - mu}")
    print(f"  原约束: x + y - 1 = {x + y - 1}  (紧约束)")
    print(f"  μ ≥ 0 ? {mu >= 0}")
    print()


# =============================================================================
# 主函数 / Main
# =============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("optimization.py — 第 13 章演示 / Chapter 13 Demo")
    print("=" * 60)
    print()
    demo_golden_section()
    trace = demo_gradient_descent()
    demo_newton_method()
    demo_bfgs()
    losses_gd, losses_sgd = demo_sgd_vs_gd()
    demo_adam()
    demo_linear_programming()
    demo_kkt_conditions()
    print("=" * 60)
    print("演示完毕 / Demo complete.")
    print("=" * 60)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        # GD 轨迹
        axes[0].plot(trace[:, 0], trace[:, 1], 'o-', markersize=3)
        axes[0].axhline(0, color='gray', lw=0.5)
        axes[0].axvline(0, color='gray', lw=0.5)
        axes[0].set_title("梯度下降轨迹 / GD trajectory")
        axes[0].grid(alpha=0.3)

        # 学习曲线
        axes[1].semilogy(losses_gd, label='Full-batch GD')
        axes[1].semilogy(losses_sgd, label='SGD (mini-batch)')
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("MSE")
        axes[1].set_title("Ch13: GD vs SGD 收敛")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig('ch13_optimization.png', dpi=120, bbox_inches='tight')
        plt.close()
        print("\n已保存可视化: ch13_optimization.png")
    except ImportError:
        print("\n(matplotlib 未安装 — 跳过画图)")
