"""
linalg_core.py — 线性代数核心模块
=================================
Module 7 of "Math is Alive" / 《数学是活的》第 7 模块

实现 / Implements:
    - 特征值与特征向量几何意义 / Eigenvalues geometry
    - 对角化 A = PDP^{-1} / Diagonalization
    - 矩阵幂 A^n 加速（对角化）/ Matrix power via diagonalization
    - Gram-Schmidt 正交化 / Gram-Schmidt orthogonalization
    - 谱定理（对称矩阵正交对角化）/ Spectral theorem
    - Pauli 矩阵（量子力学）/ Pauli matrices
    - 简化版特征脸 / Simplified eigenfaces (PCA)

Li Zhou
2026 年 6 月 / June 2026
MIT License
"""

import numpy as np


# =============================================================================
# 演示 1：特征值的几何意义 / Eigenvalue Geometry
# =============================================================================
def demo_eigenvalue_geometry():
    """A v = λ v：v 是“不被旋转”的特殊方向，λ 是缩放因子。
    Eigenvectors are directions not rotated by A.
    """
    print("\n[Demo 1] 特征值几何意义 / Eigenvalue geometry")
    A = np.array([[2, 1], [1, 2]], float)
    w, V = np.linalg.eig(A)
    for i in range(2):
        v = V[:, i]
        Av = A @ v
        print(f"  λ_{i+1} = {w[i]:+.4f},  v_{i+1} = {v.round(3)},  "
              f"A·v = {Av.round(3)}  (= λ·v ? {np.allclose(Av, w[i]*v)})")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
        fig, ax = plt.subplots(figsize=(6, 6))
        theta = np.linspace(0, 2*np.pi, 100)
        circ = np.array([np.cos(theta), np.sin(theta)])
        ell = A @ circ
        ax.plot(circ[0], circ[1], 'b--', alpha=0.5, label='单位圆')
        ax.plot(ell[0], ell[1], 'r-', label='A·单位圆 (椭圆)')
        for i in range(2):
            v = V[:, i] * w[i]
            ax.arrow(0, 0, v[0], v[1], head_width=0.1, color='green')
        ax.set_title('Ch7: 特征向量是椭圆的轴 / Eigenvectors = axes of ellipse')
        ax.axis('equal'); ax.grid(alpha=0.3); ax.legend()
        plt.tight_layout(); plt.savefig('ch07_eig.png', dpi=120); plt.close()
        print("  ✓ 已保存 ch07_eig.png")
    except ImportError:
        pass


# =============================================================================
# 演示 2：对角化 / Diagonalization
# =============================================================================
def demo_diagonalization():
    """A = P D P^{-1}。检查重构误差。
    Diagonalize A and check reconstruction.
    """
    print("\n[Demo 2] 对角化 / Diagonalization A = P D P⁻¹")
    A = np.array([[4, -2], [1, 1]], float)
    w, P = np.linalg.eig(A)
    D = np.diag(w)
    A_rec = P @ D @ np.linalg.inv(P)
    print(f"  特征值 / λ = {w}")
    print(f"  重构误差 / ||A - PDP⁻¹|| = {np.linalg.norm(A - A_rec):.3e}")


# =============================================================================
# 演示 3：矩阵幂加速 / Matrix Power via Diagonalization
# =============================================================================
def demo_matrix_power():
    """A^n = P D^n P^{-1}。对角化让幂运算 O(n^3) → O(n) 标量幂。
    Speed-up matrix power using diagonalization.
    """
    print("\n[Demo 3] 矩阵幂 / Matrix power  A^10 by diagonalization")
    A = np.array([[1, 1], [1, 0]], float)  # Fibonacci 矩阵
    w, P = np.linalg.eig(A)
    n = 10
    A_n_direct = np.linalg.matrix_power(A, n)
    A_n_diag = P @ np.diag(w**n) @ np.linalg.inv(P)
    print(f"  Fibonacci 矩阵 A^{n} (直接)：\n{A_n_direct.round(0)}")
    print(f"  通过对角化 (重构)：\n{A_n_diag.round(0)}")
    # A^n[0,1] = F_n (Fibonacci 第 n 项)
    print(f"  F_{n} = {int(round(A_n_direct[0, 1]))}  (Fibonacci)")


# =============================================================================
# 演示 4：Gram-Schmidt / Orthogonalization
# =============================================================================
def demo_gram_schmidt():
    """把 3 个线性无关向量正交化（标准）。
    Gram-Schmidt to produce orthonormal basis.
    """
    print("\n[Demo 4] Gram-Schmidt 正交化 / Orthogonalization")
    V = np.array([[1, 1, 0],
                  [1, 0, 1],
                  [0, 1, 1]], float).T  # 列是向量
    Q = np.zeros_like(V)
    for i in range(V.shape[1]):
        v = V[:, i].copy()
        for j in range(i):
            v -= (Q[:, j] @ V[:, i]) * Q[:, j]
        Q[:, i] = v / np.linalg.norm(v)
    print("  正交基 Q (列向量):\n", Q.round(4))
    print(f"  正交性 ||QᵀQ - I|| = {np.linalg.norm(Q.T @ Q - np.eye(3)):.3e}")


# =============================================================================
# 演示 5：谱定理 / Spectral Theorem
# =============================================================================
def demo_spectral_theorem():
    """对称矩阵的特征向量正交，可正交对角化 A = Q Λ Qᵀ。
    Symmetric matrices are orthogonally diagonalizable.
    """
    print("\n[Demo 5] 谱定理 / Spectral theorem (symmetric matrix)")
    A = np.array([[2, 1, 0],
                  [1, 2, 1],
                  [0, 1, 2]], float)
    w, Q = np.linalg.eigh(A)
    print(f"  对称矩阵特征值 λ = {w}")
    print(f"  Q 正交？ ||QᵀQ - I|| = {np.linalg.norm(Q.T @ Q - np.eye(3)):.3e}")
    A_rec = Q @ np.diag(w) @ Q.T
    print(f"  ||A - Q Λ Qᵀ|| = {np.linalg.norm(A - A_rec):.3e}")


# =============================================================================
# 演示 6：Pauli 矩阵 / Pauli Matrices
# =============================================================================
def demo_pauli_matrices():
    """Pauli σ_x, σ_y, σ_z 的代数性质（量子力学基础）。
    Pauli matrices: σ_i² = I, σ_x σ_y = i σ_z, etc.
    """
    print("\n[Demo 6] Pauli 矩阵 / Pauli matrices (quantum)")
    sx = np.array([[0, 1], [1, 0]], complex)
    sy = np.array([[0, -1j], [1j, 0]], complex)
    sz = np.array([[1, 0], [0, -1]], complex)
    I = np.eye(2, dtype=complex)
    for name, s in [('σ_x', sx), ('σ_y', sy), ('σ_z', sz)]:
        err = np.linalg.norm(s @ s - I)
        w, _ = np.linalg.eig(s)
        print(f"  {name}: σ² = I ? err={err:.2e},  "
              f"特征值 ≈ {sorted(w.real.round(3).tolist())}")
    print(f"  σ_x σ_y - i σ_z = 0 ? "
          f"err = {np.linalg.norm(sx @ sy - 1j*sz):.2e}")


# =============================================================================
# 演示 7：简化版特征脸 / Simplified Eigenfaces (PCA)
# =============================================================================
def demo_eigenfaces_simple():
    """生成 N 个 8x8 的简化“脸”，做 PCA，看前几个特征方向。
    Generate synthetic 8x8 faces and extract principal components.
    """
    print("\n[Demo 7] 简化版特征脸 / Simplified eigenfaces (PCA)")
    rng = np.random.default_rng(0)
    N, dim = 200, 64  # 8×8 像素
    # 生成低维结构 + 噪声
    basis = rng.standard_normal((dim, 4))
    coeffs = rng.standard_normal((N, 4))
    X = coeffs @ basis.T + 0.3*rng.standard_normal((N, dim))
    # 中心化
    Xc = X - X.mean(axis=0)
    # 协方差与特征分解
    C = (Xc.T @ Xc) / (N - 1)
    w, V = np.linalg.eigh(C)
    w, V = w[::-1], V[:, ::-1]
    explained = w / w.sum()
    print(f"  前 5 个主成分解释方差比例: "
          f"{[f'{e*100:.1f}%' for e in explained[:5]]}")
    print(f"  前 4 个主成分累积: {explained[:4].sum()*100:.1f}%  "
          "(应较高，因为数据本质 4 维)")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
        fig, axes = plt.subplots(1, 4, figsize=(10, 3))
        for i, ax in enumerate(axes):
            ax.imshow(V[:, i].reshape(8, 8), cmap='gray')
            ax.set_title(f'PC{i+1} ({explained[i]*100:.1f}%)')
            ax.axis('off')
        plt.suptitle('Ch7: 前 4 个主成分（特征脸）/ Top-4 eigenfaces')
        plt.tight_layout(); plt.savefig('ch07_eigenfaces.png', dpi=120); plt.close()
        print("  ✓ 已保存 ch07_eigenfaces.png")
    except ImportError:
        pass


# =============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("linalg_core.py — 数学是活的 / 第 7 模块演示")
    print("=" * 60)
    demo_eigenvalue_geometry()
    demo_diagonalization()
    demo_matrix_power()
    demo_gram_schmidt()
    demo_spectral_theorem()
    demo_pauli_matrices()
    demo_eigenfaces_simple()
    print("\n" + "=" * 60)
    print("演示完毕 / Demo complete.")
    print("=" * 60)
