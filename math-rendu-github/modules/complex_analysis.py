"""
complex_analysis.py — 复数分析与傅里叶
========================================
Module 9 of "Math is Alive" / 《数学是活的》第 9 章

实现 / Implements:
    - 欧拉公式 e^{iθ} = cos θ + i sin θ
    - 复数运算与几何
    - Cauchy-Riemann 方程
    - FFT 与音频频谱
    - 留数定理示例
    - MP3 风格的有损压缩

数学基础 / Math:
    e^{iπ} + 1 = 0
    ∮ f dz = 2πi · Σ Res(f, z_k)
    FFT: O(N log N) 的 DFT

Li Zhou
2027 / MIT License
"""

import numpy as np


# =============================================================================
# 演示 1：欧拉公式
# Demo 1: Euler's formula
# =============================================================================
def demo_euler_formula():
    """e^{iθ} 在复平面上画出单位圆。
    e^{iθ} traces the unit circle in the complex plane.
    """
    print("【演示 1】欧拉公式 e^{iθ} = cos θ + i sin θ")
    print("[Demo 1] Euler's formula")
    print()
    for theta_name, theta in [("0", 0),
                              ("π/4", np.pi / 4),
                              ("π/2", np.pi / 2),
                              ("π",   np.pi)]:
        z = np.exp(1j * theta)
        print(f"  θ = {theta_name:>4s}: e^(iθ) = "
              f"{z.real:+.3f} + {z.imag:+.3f}i")
    print("  ⇒ θ=π 给出 e^(iπ) = -1 ⇒ 欧拉恒等式 e^{iπ}+1=0")
    print()


# =============================================================================
# 演示 2：复数运算 (旋转 = 复数乘法)
# Demo 2: Complex arithmetic (rotation = multiplication)
# =============================================================================
def demo_complex_arithmetic():
    """乘以 e^{iα} 等价于旋转角度 α。
    Multiplying by e^{iα} is rotation by α.
    """
    print("【演示 2】复数乘法 = 旋转 + 缩放")
    print("[Demo 2] Complex multiplication = rotation + scaling")
    print()
    z = 1 + 1j
    print(f"  原点 z = {z} (角度 {np.angle(z, deg=True):.1f}°, "
          f"模 {abs(z):.3f})")
    for deg in [30, 90, 180]:
        w = np.exp(1j * np.deg2rad(deg))
        zr = z * w
        print(f"  旋转 {deg:>3d}°: → "
              f"{zr.real:+.3f} + {zr.imag:+.3f}i  "
              f"(角度 {np.angle(zr, deg=True):.1f}°)")
    print()


# =============================================================================
# 演示 3：Cauchy-Riemann 方程
# Demo 3: Cauchy-Riemann equations
# =============================================================================
def demo_cauchy_riemann():
    """全纯函数必须满足 u_x = v_y, u_y = -v_x。
    Holomorphic functions satisfy u_x = v_y, u_y = -v_x.
    """
    print("【演示 3】Cauchy-Riemann 方程")
    print("[Demo 3] Cauchy-Riemann equations")
    print()
    # 取 f(z) = z² = (x+iy)² = (x²-y²) + i(2xy)
    # u = x²-y², v = 2xy
    # u_x = 2x, v_y = 2x  ✓
    # u_y = -2y, v_x = 2y ⇒ u_y = -v_x ✓
    x, y = 1.5, 0.7
    u_x = 2 * x
    v_y = 2 * x
    u_y = -2 * y
    v_x = 2 * y
    print(f"  f(z) = z² 在 (x={x}, y={y})")
    print(f"  u_x = {u_x},   v_y = {v_y}   相等? {u_x == v_y}")
    print(f"  u_y = {u_y},  -v_x = {-v_x}  相等? {u_y == -v_x}")
    print("  ⇒ z² 是全纯函数 / z² is holomorphic")
    print()


# =============================================================================
# 演示 4：FFT —— 找出信号中的频率
# Demo 4: FFT — find frequencies in a signal
# =============================================================================
def demo_fft_audio():
    """合成两个正弦波，再用 FFT 分解出来。
    Compose two sines, then decompose them via FFT.
    """
    print("【演示 4】FFT 频谱分析")
    print("[Demo 4] FFT spectrum analysis")
    print()
    fs = 1000  # 采样率 / sampling rate
    T = 1.0    # 时长
    t = np.linspace(0, T, int(fs * T), endpoint=False)
    f1, f2 = 50, 120
    x = 1.0 * np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1 / fs)
    magnitude = np.abs(X) / len(x) * 2
    # 找前两个峰
    peaks = np.argsort(-magnitude)[:2]
    print(f"  合成频率 / synthesized: {f1} Hz (幅 1.0), "
          f"{f2} Hz (幅 0.5)")
    for p in sorted(peaks):
        print(f"  FFT 检测到峰: f = {freqs[p]:.1f} Hz, "
              f"幅度 ≈ {magnitude[p]:.3f}")
    print()
    return t, x, freqs, magnitude


# =============================================================================
# 演示 5：留数定理
# Demo 5: Residue theorem
# =============================================================================
def demo_residue():
    """∮ 1/(z²+1) dz 沿大圆 = 2πi · 留数和。
    ∮ 1/(z²+1) dz along a big circle = 2πi · sum of residues.
    """
    print("【演示 5】留数定理")
    print("[Demo 5] Residue theorem")
    print()
    # f(z) = 1/(z² + 1) 极点在 ±i
    # Res(f, +i) = 1/(2i), Res(f, -i) = -1/(2i)
    # 若回路只绕 +i，积分 = 2πi · 1/(2i) = π
    R = 5.0  # 大半圆半径
    # 数值积分沿圆 |z|=R 的上半 + 实轴段，逼近 π
    N = 5000
    theta = np.linspace(0, 2 * np.pi, N)
    z = R * np.exp(1j * theta)
    dz = 1j * R * np.exp(1j * theta) * (2 * np.pi / N)
    integral = np.sum(1.0 / (z ** 2 + 1) * dz)
    expected = 2j * np.pi * (1.0 / (2j) + (-1.0 / (2j)))  # = 0
    # 因为绕两个极点，总和 = 0
    print(f"  数值 ∮ 1/(z²+1) dz, |z|=R ≈ {integral:.4f}")
    print(f"  理论 (绕 +i 和 -i 各一次): {expected:.4f}")
    print(f"  误差: {abs(integral - expected):.2e}")
    print()


# =============================================================================
# 演示 6：MP3 风格的有损压缩
# Demo 6: MP3-like lossy compression
# =============================================================================
def demo_mp3_compression():
    """在 FFT 域里把小幅度的频率分量置零，再反变换。
    Zero out low-amplitude frequency bins, inverse-FFT.
    """
    print("【演示 6】MP3 风格压缩")
    print("[Demo 6] MP3-like compression")
    print()
    fs = 2000
    t = np.linspace(0, 1, fs, endpoint=False)
    x = (np.sin(2 * np.pi * 100 * t)
         + 0.7 * np.sin(2 * np.pi * 250 * t)
         + 0.2 * np.sin(2 * np.pi * 500 * t)
         + 0.1 * np.random.default_rng(0).standard_normal(fs))
    X = np.fft.rfft(x)
    for keep_frac in [1.0, 0.1, 0.05, 0.02]:
        Xk = X.copy()
        threshold = np.quantile(np.abs(Xk), 1 - keep_frac)
        Xk[np.abs(Xk) < threshold] = 0
        x_recon = np.fft.irfft(Xk, n=len(x))
        snr = 10 * np.log10(np.sum(x ** 2)
                            / np.sum((x - x_recon) ** 2 + 1e-12))
        print(f"  保留 {keep_frac*100:5.1f}% 频率分量 "
              f"→ SNR = {snr:5.1f} dB")
    print("  ⇒ MP3 用心理声学模型决定丢哪些频率")
    print()


# =============================================================================
# 主函数 / Main
# =============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("complex_analysis.py — 第 9 章演示 / Chapter 9 Demo")
    print("=" * 60)
    print()
    demo_euler_formula()
    demo_complex_arithmetic()
    demo_cauchy_riemann()
    t, x, freqs, magnitude = demo_fft_audio()
    demo_residue()
    demo_mp3_compression()
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
        # 单位圆
        theta = np.linspace(0, 2 * np.pi, 200)
        axes[0].plot(np.cos(theta), np.sin(theta))
        axes[0].plot([0, 1], [0, 0], 'r-', lw=2)
        axes[0].set_title("欧拉公式 / 单位圆")
        axes[0].axis('equal')
        axes[0].grid(alpha=0.3)

        # 时域信号
        axes[1].plot(t[:200], x[:200])
        axes[1].set_title("时域信号 / Time domain")
        axes[1].set_xlabel("t (s)")

        # 频谱
        axes[2].plot(freqs, magnitude)
        axes[2].set_xlim(0, 200)
        axes[2].set_title("Ch9: FFT 频谱 / Spectrum")
        axes[2].set_xlabel("f (Hz)")
        axes[2].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig('ch09_complex.png', dpi=120, bbox_inches='tight')
        plt.close()
        print("\n已保存可视化: ch09_complex.png")
    except ImportError:
        print("\n(matplotlib 未安装 — 跳过画图)")
