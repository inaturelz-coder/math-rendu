"""
pde.py — 偏微分方程模块
=======================
Module 5 of "Math is Alive" / 《数学是活的》第 5 模块

实现 / Implements:
    - PDE 分类（抛物 / 双曲 / 椭圆）/ Classification
    - 一维热方程 u_t = α u_xx (FDM 显式) / 1D heat equation
    - 一维波方程 u_tt = c² u_xx / 1D wave equation
    - 二维 Laplace ∇²u = 0 (Jacobi 迭代) / 2D Laplace
    - 芯片散热（带源项的热方程）/ Chip heat dissipation
    - Fourier 级数：方波展开 / Fourier series of square wave

Li Zhou
2026 年 6 月 / June 2026
MIT License
"""

import numpy as np


# =============================================================================
# 演示 1：PDE 分类 / Classification
# =============================================================================
def demo_classify():
    """对 A u_xx + 2B u_xy + C u_yy + ... = 0 用判别式 B² - AC。
    Discriminant > 0 双曲 / = 0 抛物 / < 0 椭圆.
    """
    print("\n[Demo 1] PDE 二阶分类 / Second-order PDE classification")
    cases = [
        ('热方程  u_t = α u_xx',  1, 0, 0, '抛物 / parabolic'),
        ('波方程  u_tt = c² u_xx', 1, 0, -1, '双曲 / hyperbolic'),
        ('Laplace ∇²u = 0',        1, 0, 1, '椭圆 / elliptic'),
    ]
    for name, A, B, C, label in cases:
        disc = B*B - A*C
        print(f"  {name:25s}  B²-AC = {disc:+}   →  {label}")


# =============================================================================
# 演示 2：一维热方程 / 1D Heat
# =============================================================================
def demo_heat_1d():
    """u_t = α u_xx, 端点固定 u=0，初始 u(x,0)=sin(πx)。
    Explicit FDM (稳定条件 r = α dt/dx² ≤ 0.5).
    """
    print("\n[Demo 2] 一维热方程 / 1D heat equation (FDM explicit)")
    alpha, L, N = 0.01, 1.0, 50
    dx = L/N
    dt = 0.4 * dx**2 / alpha
    r = alpha*dt/dx**2
    print(f"  α={alpha}  dx={dx:.4f}  dt={dt:.5f}  r={r:.3f} (≤0.5 稳定)")
    x = np.linspace(0, L, N+1)
    u = np.sin(np.pi*x)
    snapshots = [u.copy()]
    for step in range(1, 2001):
        u_new = u.copy()
        u_new[1:-1] = u[1:-1] + r*(u[2:] - 2*u[1:-1] + u[:-2])
        u = u_new
        if step in (200, 600, 2000):
            snapshots.append(u.copy())
    print(f"  最大值 / Max u: t=0 → {snapshots[0].max():.3f},  "
          f"末态 → {snapshots[-1].max():.3e}  (扩散衰减)")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
        plt.figure(figsize=(8, 4))
        labels = ['t=0', 't=200·dt', 't=600·dt', 't=2000·dt']
        for s, lab in zip(snapshots, labels):
            plt.plot(x, s, label=lab)
        plt.title('Ch5: 一维热扩散 / 1D heat diffusion')
        plt.xlabel('x'); plt.ylabel('u(x,t)')
        plt.legend(); plt.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig('ch05_heat.png', dpi=120); plt.close()
        print("  ✓ 已保存 ch05_heat.png")
    except ImportError:
        pass


# =============================================================================
# 演示 3：一维波方程 / 1D Wave
# =============================================================================
def demo_wave_1d():
    """u_tt = c² u_xx, 固定端点，弦的振动。
    1D wave equation, fixed ends, leapfrog scheme.
    """
    print("\n[Demo 3] 一维波方程 / 1D wave equation")
    c, L, N = 1.0, 1.0, 100
    dx = L/N
    dt = 0.5 * dx / c
    cfl = c*dt/dx
    print(f"  c={c}  dx={dx:.4f}  dt={dt:.5f}  CFL={cfl:.3f} (≤1)")
    x = np.linspace(0, L, N+1)
    u_prev = np.sin(np.pi*x)
    u = u_prev.copy()  # 初始速度 = 0
    snapshots = [u.copy()]
    for step in range(1, 401):
        u_new = np.zeros_like(u)
        u_new[1:-1] = (2*u[1:-1] - u_prev[1:-1]
                       + cfl**2*(u[2:] - 2*u[1:-1] + u[:-2]))
        u_prev = u
        u = u_new
        if step in (50, 100, 200, 400):
            snapshots.append(u.copy())
    print("  弦在反复振动，能量守恒（数值上接近）")
    print("  String oscillates; energy approximately conserved")


# =============================================================================
# 演示 4：二维 Laplace / 2D Laplace
# =============================================================================
def demo_laplace_2d():
    """∇²u = 0 on [0,1]² with u=0 三边、u=1 顶边. Jacobi 迭代。
    Solve Laplace equation with Dirichlet BC by Jacobi iteration.
    """
    print("\n[Demo 4] 2D Laplace ∇²u=0   (Jacobi)")
    N = 50
    u = np.zeros((N, N))
    u[-1, :] = 1.0  # 顶边
    for it in range(5000):
        u_new = u.copy()
        u_new[1:-1, 1:-1] = 0.25*(u[2:, 1:-1] + u[:-2, 1:-1]
                                  + u[1:-1, 2:] + u[1:-1, :-2])
        if np.max(np.abs(u_new - u)) < 1e-6:
            print(f"  收敛 / Converged at iter {it}")
            u = u_new
            break
        u = u_new
    print(f"  中心 u(N/2, N/2) ≈ {u[N//2, N//2]:.4f}  (理论 ≈ 0.25)")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
        plt.figure(figsize=(6, 5))
        plt.imshow(u, origin='lower', cmap='hot')
        plt.colorbar(label='u')
        plt.title('Ch5: 2D Laplace 稳态解 / steady-state')
        plt.tight_layout(); plt.savefig('ch05_laplace.png', dpi=120); plt.close()
        print("  ✓ 已保存 ch05_laplace.png")
    except ImportError:
        pass


# =============================================================================
# 演示 5：芯片散热 / Chip Heat
# =============================================================================
def demo_chip_heat():
    """带源项的稳态热方程：-∇²T = q(x,y)。CPU 局部高功率源。
    Steady-state heat with heat source — Poisson equation.
    """
    print("\n[Demo 5] CPU 芯片散热 / Chip steady-state heat (Poisson)")
    N = 60
    T = np.zeros((N, N))
    # 热源：中央 8x8 区域
    q = np.zeros((N, N))
    cx, cy = N//2, N//2
    q[cx-4:cx+4, cy-4:cy+4] = 50.0
    dx = 1.0/N
    for it in range(8000):
        T_new = T.copy()
        T_new[1:-1, 1:-1] = 0.25*(T[2:, 1:-1] + T[:-2, 1:-1]
                                  + T[1:-1, 2:] + T[1:-1, :-2]
                                  + dx*dx*q[1:-1, 1:-1])
        if np.max(np.abs(T_new - T)) < 1e-6:
            print(f"  收敛 / Converged at iter {it}")
            T = T_new
            break
        T = T_new
    print(f"  最高温度 / Max T = {T.max():.3f}  at center")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
        plt.figure(figsize=(6, 5))
        plt.imshow(T, origin='lower', cmap='inferno')
        plt.colorbar(label='温度 T')
        plt.title('Ch5: 芯片稳态温度 / Chip steady T')
        plt.tight_layout(); plt.savefig('ch05_chip.png', dpi=120); plt.close()
        print("  ✓ 已保存 ch05_chip.png")
    except ImportError:
        pass


# =============================================================================
# 演示 6：Fourier 级数 / Fourier Series
# =============================================================================
def demo_fourier_series():
    """方波的 Fourier 级数展开。Gibbs 现象。
    Square wave Fourier expansion, Gibbs phenomenon.
    """
    print("\n[Demo 6] Fourier 级数 / Square wave expansion")
    x = np.linspace(-np.pi, np.pi, 1000)
    for K in [1, 5, 21, 101]:
        s = np.zeros_like(x)
        for k in range(1, K+1, 2):  # 奇数项
            s += (4/np.pi) * np.sin(k*x)/k
        err = np.max(np.abs(s)) - 1.0  # Gibbs overshoot
        print(f"  K={K:>4}  最大值 ≈ {np.max(s):.4f}  (overshoot {err:+.4f})")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
        plt.figure(figsize=(8, 4))
        sq = np.sign(np.sin(x))
        plt.plot(x, sq, 'k--', label='方波 / Square')
        for K in [1, 5, 21]:
            s = np.zeros_like(x)
            for k in range(1, K+1, 2):
                s += (4/np.pi) * np.sin(k*x)/k
            plt.plot(x, s, label=f'K={K}')
        plt.title('Ch5: 方波 Fourier 展开 / Square wave Fourier')
        plt.legend(); plt.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig('ch05_fourier.png', dpi=120); plt.close()
        print("  ✓ 已保存 ch05_fourier.png")
    except ImportError:
        pass


# =============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("pde.py — 数学是活的 / 第 5 模块演示")
    print("=" * 60)
    demo_classify()
    demo_heat_1d()
    demo_wave_1d()
    demo_laplace_2d()
    demo_chip_heat()
    demo_fourier_series()
    print("\n" + "=" * 60)
    print("演示完毕 / Demo complete.")
    print("=" * 60)
