"""
multi_calc.py — 多元微积分模块
==============================
Module 2 of "Math is Alive" / 《数学是活的》第 2 模块

实现 / Implements:
    - 偏导数 / Partial derivatives
    - 梯度场可视化 / Gradient field visualization
    - 全微分线性近似 / Total differential linearization
    - 拉格朗日乘数法（解析 & 数值）/ Lagrange multipliers
    - 送餐路径优化（约束最优化应用）/ Delivery route optimization
    - 二重积分（trapz 二维）/ Double integrals

Li Zhou
2026 年 6 月 / June 2026
MIT License
"""

import numpy as np


# =============================================================================
# 演示 1：偏导数 / Partial Derivatives
# =============================================================================
def demo_partial_derivatives():
    """计算 f(x,y) = x²y + sin(xy) 的 ∂f/∂x 和 ∂f/∂y 在 (1, π/4)。
    Compute partial derivatives at (1, π/4).
    """
    print("\n[Demo 1] 偏导数 / Partial derivatives of f(x,y)=x²y + sin(xy)")
    x0, y0 = 1.0, np.pi/4
    # 解析 / analytical
    fx_a = 2*x0*y0 + y0*np.cos(x0*y0)
    fy_a = x0**2 + x0*np.cos(x0*y0)
    # 数值 / numerical
    h = 1e-5
    f = lambda x, y: x**2*y + np.sin(x*y)
    fx_n = (f(x0+h, y0) - f(x0-h, y0))/(2*h)
    fy_n = (f(x0, y0+h) - f(x0, y0-h))/(2*h)
    print(f"  ∂f/∂x  解析={fx_a:.6f}  数值={fx_n:.6f}")
    print(f"  ∂f/∂y  解析={fy_a:.6f}  数值={fy_n:.6f}")


# =============================================================================
# 演示 2：梯度场 / Gradient Field
# =============================================================================
def demo_gradient_field():
    """画 f(x,y) = x² + y² 的梯度场（应指向远离原点）。
    Plot gradient field of f(x,y) = x²+y² (should point away from origin).
    """
    print("\n[Demo 2] 梯度场 / Gradient field of x²+y²")
    x = np.linspace(-2, 2, 20)
    y = np.linspace(-2, 2, 20)
    X, Y = np.meshgrid(x, y)
    U, V = 2*X, 2*Y
    print(f"  在 (1,1)  ∇f = ({U[len(y)//2 + 2, len(x)//2+2]:.1f}, "
          f"{V[len(y)//2+2, len(x)//2+2]:.1f})")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
        plt.figure(figsize=(6, 6))
        Z = X**2 + Y**2
        plt.contour(X, Y, Z, levels=10, alpha=0.6)
        plt.quiver(X, Y, U, V, color='red', alpha=0.7)
        plt.title('Ch2: 梯度场 ∇(x²+y²) / Gradient Field')
        plt.xlabel('x'); plt.ylabel('y'); plt.axis('equal')
        plt.tight_layout()
        plt.savefig('ch02_grad_field.png', dpi=120)
        plt.close()
        print("  ✓ 已保存 ch02_grad_field.png")
    except ImportError:
        pass


# =============================================================================
# 演示 3：全微分 / Total Differential
# =============================================================================
def demo_total_differential():
    """全微分 df ≈ f_x dx + f_y dy，验证线性近似的精度。
    Total differential as linear approximation.
    """
    print("\n[Demo 3] 全微分线性近似 / Total differential linearization")
    f = lambda x, y: x**2 + 3*x*y + y**3
    x0, y0 = 1.0, 1.0
    f0 = f(x0, y0)
    fx, fy = 2*x0 + 3*y0, 3*x0 + 3*y0**2

    for dx, dy in [(0.1, 0.1), (0.01, 0.01), (0.001, 0.001)]:
        df_lin = fx*dx + fy*dy
        df_act = f(x0+dx, y0+dy) - f0
        err = abs(df_act - df_lin)
        print(f"  (dx,dy)=({dx},{dy})  线性={df_lin:.6f}  "
              f"实际={df_act:.6f}  误差={err:.2e}")
    print("  → 误差 ~ O((dx,dy)²) / Error is second order")


# =============================================================================
# 演示 4：拉格朗日乘数 / Lagrange Multipliers
# =============================================================================
def demo_lagrange():
    """在 x² + y² = 1 上极大化 f(x,y)=xy。最大值 = 1/2，在 (±1/√2, ±1/√2)。
    Maximize f(x,y)=xy on unit circle. Max = 0.5 at ±(1/√2, 1/√2).
    """
    print("\n[Demo 4] 拉格朗日乘数 / Lagrange max of xy on x²+y²=1")
    # 用 numpy.linalg 解小线性问题：∇f = λ ∇g
    # f_x = y = 2λx ; f_y = x = 2λy  →  y/x = x/y → x² = y², 代入 x²+y²=1
    # 解：x = ±1/√2, y = ±1/√2, 极值 = ±0.5
    pts = [(1/np.sqrt(2), 1/np.sqrt(2)),
           (-1/np.sqrt(2), -1/np.sqrt(2)),
           (1/np.sqrt(2), -1/np.sqrt(2)),
           (-1/np.sqrt(2), 1/np.sqrt(2))]
    for x, y in pts:
        print(f"  ({x:+.4f}, {y:+.4f})  →  xy = {x*y:+.4f}")
    print("  最大 / max = 0.5,  最小 / min = -0.5")


# =============================================================================
# 演示 5：送餐路径优化 / Delivery Optimization
# =============================================================================
def demo_delivery_optimization():
    """一个外卖员要在 3 家餐厅取餐，总路径长度最短（简化版：取餐顺序 + 起终点）。
    Delivery man visiting 3 restaurants—minimize total path length.
    """
    print("\n[Demo 5] 送餐路径优化 / Delivery shortest path")
    # 起点、餐厅、终点（顾客）
    start = np.array([0, 0])
    end = np.array([5, 5])
    restaurants = np.array([[1, 4], [3, 1], [4, 3]])
    from itertools import permutations
    best, best_len = None, np.inf
    for perm in permutations(range(3)):
        path = [start, *restaurants[list(perm)], end]
        total = sum(np.linalg.norm(path[i+1]-path[i]) for i in range(len(path)-1))
        if total < best_len:
            best, best_len = perm, total
    print(f"  最佳顺序 / Best order: {best}")
    print(f"  最短距离 / Min distance: {best_len:.4f}")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
        plt.figure(figsize=(6, 6))
        path = np.array([start, *restaurants[list(best)], end])
        plt.plot(path[:, 0], path[:, 1], 'o-', color='steelblue')
        plt.scatter(*start, color='green', s=120, label='起点')
        plt.scatter(*end, color='red', s=120, label='终点')
        plt.scatter(restaurants[:, 0], restaurants[:, 1], color='orange',
                    s=80, label='餐厅')
        plt.title(f'Ch2: 送餐最短路径 / Delivery  (L={best_len:.2f})')
        plt.legend(); plt.grid(alpha=0.3); plt.axis('equal')
        plt.tight_layout()
        plt.savefig('ch02_delivery.png', dpi=120)
        plt.close()
        print("  ✓ 已保存 ch02_delivery.png")
    except ImportError:
        pass


# =============================================================================
# 演示 6：二重积分 / Double Integral
# =============================================================================
def demo_double_integral():
    """∬_{[0,1]²} (x² + y²) dxdy = 2/3 ≈ 0.6667。
    Double integral of (x²+y²) over [0,1]² = 2/3.
    """
    print("\n[Demo 6] 二重积分 / Double integral  ∬(x²+y²) dxdy over [0,1]²")
    print("         真值 / True = 2/3 ≈ 0.6667")
    for N in [10, 50, 200]:
        x = np.linspace(0, 1, N)
        y = np.linspace(0, 1, N)
        X, Y = np.meshgrid(x, y)
        Z = X**2 + Y**2
        I = np.trapz(np.trapz(Z, y, axis=0), x)
        print(f"  N={N:>4}  I = {I:.6f}")


# =============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("multi_calc.py — 数学是活的 / 第 2 模块演示")
    print("=" * 60)
    demo_partial_derivatives()
    demo_gradient_field()
    demo_total_differential()
    demo_lagrange()
    demo_delivery_optimization()
    demo_double_integral()
    print("\n" + "=" * 60)
    print("演示完毕 / Demo complete.")
    print("=" * 60)
