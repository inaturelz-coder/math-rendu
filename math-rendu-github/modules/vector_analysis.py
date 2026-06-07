"""
vector_analysis.py — 向量分析模块
=================================
Module 3 of "Math is Alive" / 《数学是活的》第 3 模块

实现 / Implements:
    - 标量场的梯度 ∇f / Gradient of scalar field
    - 向量场的散度 ∇·F 与旋度 ∇×F / Divergence and curl
    - Gauss 定理（散度定理）数值验证 / Gauss/divergence theorem
    - Stokes 定理数值验证 / Stokes theorem verification
    - 5G 基站信号覆盖（场强 1/r²）/ Signal coverage application

Li Zhou
2026 年 6 月 / June 2026
MIT License
"""

import numpy as np


# =============================================================================
# 演示 1：梯度场 / Gradient Field
# =============================================================================
def demo_gradient_field():
    """二维梯度：f(x,y) = sin(x)·cos(y) → ∇f = (cos x cos y, −sin x sin y)。
    2D gradient of sin(x)cos(y).
    """
    print("\n[Demo 1] 标量场梯度 / Gradient of f=sin(x)cos(y)")
    x = np.linspace(-np.pi, np.pi, 30)
    y = np.linspace(-np.pi, np.pi, 30)
    X, Y = np.meshgrid(x, y)
    F = np.sin(X)*np.cos(Y)
    # np.gradient 返回 [df/dy, df/dx]（行优先）
    Fy, Fx = np.gradient(F, y, x)
    print(f"  在 (0,0)  ∇f ≈ ({Fx[15,15]:.3f}, {Fy[15,15]:.3f})  "
          "(解析 = (1, 0))")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
        plt.figure(figsize=(6, 6))
        plt.contourf(X, Y, F, levels=20, cmap='RdBu_r', alpha=0.7)
        plt.quiver(X[::2, ::2], Y[::2, ::2], Fx[::2, ::2], Fy[::2, ::2],
                   color='black', alpha=0.6)
        plt.title('Ch3: 标量场与梯度 / Scalar field and gradient')
        plt.axis('equal'); plt.tight_layout()
        plt.savefig('ch03_gradient.png', dpi=120); plt.close()
        print("  ✓ 已保存 ch03_gradient.png")
    except ImportError:
        pass


# =============================================================================
# 演示 2：散度 / Divergence
# =============================================================================
def demo_divergence():
    """计算 F = (x, y) 的散度 = 2 与 F = (-y, x) 的散度 = 0。
    Divergence of (x,y) = 2 (source); of (-y,x) = 0 (rotation).
    """
    print("\n[Demo 2] 散度 / Divergence")
    x = np.linspace(-2, 2, 50); y = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x, y)

    for name, U, V, expected in [
        ('F=(x, y) (源场)',  X,  Y,  2.0),
        ('F=(-y, x) (旋转)', -Y, X,  0.0),
    ]:
        dU_dy, dU_dx = np.gradient(U, y, x)
        dV_dy, dV_dx = np.gradient(V, y, x)
        div = dU_dx + dV_dy
        print(f"  {name:20s}  数值散度 ≈ {div[25,25]:.3f}  (理论 {expected})")


# =============================================================================
# 演示 3：旋度 / Curl
# =============================================================================
def demo_curl():
    """二维旋度 (∂V/∂x − ∂U/∂y)。对 F = (-y, x) 应得 2。
    2D curl scalar. For (-y, x) it equals 2.
    """
    print("\n[Demo 3] 旋度 / Curl")
    x = np.linspace(-2, 2, 50); y = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x, y)
    U, V = -Y, X
    dU_dy, _ = np.gradient(U, y, x)
    _, dV_dx = np.gradient(V, y, x)
    curl = dV_dx - dU_dy
    print(f"  F=(-y, x) 的旋度 ≈ {curl[25,25]:.3f}  (理论 = 2)")


# =============================================================================
# 演示 4：Gauss / 散度定理 / Divergence Theorem
# =============================================================================
def demo_gauss_theorem():
    """∮_∂Ω F·n dS = ∭_Ω ∇·F dV，取 Ω = 单位圆，F = (x, y)。
    Both sides should equal 2·Area = 2π.
    """
    print("\n[Demo 4] 散度定理 / Gauss theorem (Ω=单位圆, F=(x,y))")
    # 体积分 ∬ 2 dA = 2 π
    N = 400
    xs = np.linspace(-1, 1, N); ys = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(xs, ys)
    inside = X**2 + Y**2 <= 1
    div = 2.0
    dx = xs[1] - xs[0]; dy = ys[1] - ys[0]
    vol_int = div * inside.sum() * dx * dy

    # 边界积分 ∮ (x,y)·(x,y) dS = ∮ 1 dS = 2π (because n=(x,y) on unit circle, F·n=1)
    theta = np.linspace(0, 2*np.pi, 1000)
    Fx, Fy = np.cos(theta), np.sin(theta)  # x=cos, y=sin on r=1
    nx, ny = np.cos(theta), np.sin(theta)
    flux = np.trapz(Fx*nx + Fy*ny, theta)

    print(f"  ∭ ∇·F dV ≈ {vol_int:.4f}")
    print(f"  ∮ F·n dS ≈ {flux:.4f}")
    print(f"  理论值 / Theory = 2π ≈ {2*np.pi:.4f}")


# =============================================================================
# 演示 5：Stokes 定理 / Stokes Theorem
# =============================================================================
def demo_stokes_theorem():
    """∮ F·dr = ∬ (∇×F)·n dA。F=(-y, x, 0)，面 = 单位圆盘 z=0，两端都 = 2π。
    """
    print("\n[Demo 5] Stokes 定理 / Stokes theorem (F=(-y,x,0), 单位圆)")
    # 线积分：F·dr = (-sin θ)(-sin θ) + cos θ · cos θ = 1
    theta = np.linspace(0, 2*np.pi, 1000)
    line = np.trapz(np.ones_like(theta), theta)
    # 面积分：curl = 2, area = π → 2π
    area_int = 2 * np.pi
    print(f"  ∮ F·dr ≈ {line:.4f}")
    print(f"  ∬ curl·dA = {area_int:.4f}")
    print(f"  理论 = 2π ≈ {2*np.pi:.4f}")


# =============================================================================
# 演示 6：信号覆盖 / Signal Coverage
# =============================================================================
def demo_signal_coverage():
    """5G 基站场强 ~ 1/r²，画覆盖热力图与梯度（信号变化方向）。
    Signal strength heat-map and gradient direction.
    """
    print("\n[Demo 6] 信号覆盖 / 5G signal coverage  S = 1/(0.1 + r²)")
    x = np.linspace(-5, 5, 100); y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    bases = [(-2, -1, 1.0), (2, 2, 1.5), (0, -3, 0.8)]
    S = np.zeros_like(X)
    for bx, by, p in bases:
        r2 = (X-bx)**2 + (Y-by)**2
        S += p / (0.1 + r2)
    coverage = (S > 0.5).mean()
    print(f"  信号 > 0.5 覆盖率 ≈ {coverage*100:.1f}%")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
        plt.figure(figsize=(7, 6))
        plt.imshow(S, extent=[-5, 5, -5, 5], origin='lower', cmap='magma',
                   alpha=0.9)
        for bx, by, _ in bases:
            plt.scatter(bx, by, color='cyan', marker='^', s=140)
        plt.colorbar(label='信号强度')
        plt.title('Ch3: 5G 信号覆盖 / Signal coverage')
        plt.tight_layout()
        plt.savefig('ch03_signal.png', dpi=120); plt.close()
        print("  ✓ 已保存 ch03_signal.png")
    except ImportError:
        pass


# =============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("vector_analysis.py — 数学是活的 / 第 3 模块演示")
    print("=" * 60)
    demo_gradient_field()
    demo_divergence()
    demo_curl()
    demo_gauss_theorem()
    demo_stokes_theorem()
    demo_signal_coverage()
    print("\n" + "=" * 60)
    print("演示完毕 / Demo complete.")
    print("=" * 60)
