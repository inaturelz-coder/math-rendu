"""
linalg_apps.py — 线性代数的应用
==================================
Module 8 of "Math is Alive" / 《数学是活的》第 8 章

实现 / Implements:
    - SVD 分解与图像压缩
    - PCA 主成分分析
    - 最小二乘法回归
    - PageRank 算法
    - LU 与 QR 分解
    - 神经网络一层的矩阵实现

数学基础 / Math:
    A = U Σ V^T            (SVD)
    PCA = SVD on centered data
    x* = (A^T A)^{-1} A^T b (Least squares)
    PageRank: x = α M x + (1-α)/n · 1

Li Zhou
2027 / MIT License
"""

import numpy as np
from numpy.linalg import svd, qr, lstsq, eig


# =============================================================================
# 演示 1：SVD —— 把矩阵切成三块
# Demo 1: SVD — slice a matrix into three pieces
# =============================================================================
def demo_svd():
    """SVD 分解的基本用法和几何含义。
    Basic usage and geometric meaning of SVD.
    """
    print("【演示 1】SVD 奇异值分解")
    print("[Demo 1] Singular Value Decomposition")
    print()
    A = np.array([[3, 1, 1],
                  [-1, 3, 1]], dtype=float)
    U, S, Vt = svd(A, full_matrices=False)
    print(f"  原矩阵 A =\n{A}")
    print(f"  U =\n{U}")
    print(f"  Σ (奇异值) = {S}")
    print(f"  V^T =\n{Vt}")
    A_recon = U @ np.diag(S) @ Vt
    print(f"  重建误差 / Reconstruction err: {np.linalg.norm(A - A_recon):.2e}")
    print()


# =============================================================================
# 演示 2：PCA —— 找数据的主轴
# Demo 2: PCA — find the principal axes
# =============================================================================
def demo_pca():
    """PCA 通过 SVD 实现：在最大方差方向降维。
    PCA via SVD: project onto directions of maximum variance.
    """
    print("【演示 2】PCA 主成分分析")
    print("[Demo 2] Principal Component Analysis")
    print()
    rng = np.random.default_rng(42)
    # 二维相关数据 / correlated 2D data
    X = rng.standard_normal((200, 2)) @ np.array([[2.0, 1.0], [1.0, 0.8]])
    Xc = X - X.mean(axis=0)
    U, S, Vt = svd(Xc, full_matrices=False)
    variance = (S ** 2) / (len(X) - 1)
    explained = variance / variance.sum()
    print(f"  数据形状 / shape: {X.shape}")
    print(f"  主成分方向 / Principal axes:\n{Vt}")
    print(f"  方差占比 / Explained variance: {explained}")
    print(f"  ⇒ 第一主成分解释 {explained[0]*100:.1f}% 方差")
    print()
    return X, Vt


# =============================================================================
# 演示 3：最小二乘 —— 把点拟合到直线
# Demo 3: Least squares — fit points to a line
# =============================================================================
def demo_least_squares():
    """y = a + b x 的最小二乘拟合。
    Least-squares fit y = a + b x.
    """
    print("【演示 3】最小二乘拟合")
    print("[Demo 3] Least squares fitting")
    print()
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 30)
    y_true = 2.0 + 0.5 * x
    y = y_true + rng.normal(scale=0.5, size=x.shape)
    A = np.column_stack([np.ones_like(x), x])
    coef, *_ = lstsq(A, y, rcond=None)
    print(f"  真实系数 / True: a=2.0,  b=0.5")
    print(f"  拟合系数 / Fit:  a={coef[0]:.3f}, b={coef[1]:.3f}")
    print(f"  残差范数 / Residual norm: "
          f"{np.linalg.norm(y - A @ coef):.3f}")
    print()
    return x, y, coef


# =============================================================================
# 演示 4：PageRank —— Google 的算法骨架
# Demo 4: PageRank — Google's algorithmic backbone
# =============================================================================
def demo_pagerank():
    """简单 4 节点网络上的 PageRank 迭代。
    PageRank iteration on a 4-node toy web.
    """
    print("【演示 4】PageRank")
    print("[Demo 4] PageRank")
    print()
    # 链接矩阵：L[i,j] = 1 表示 j 指向 i
    L = np.array([[0, 0, 1, 1/2],
                  [1/3, 0, 0, 0],
                  [1/3, 1/2, 0, 1/2],
                  [1/3, 1/2, 0, 0]])
    n = L.shape[0]
    alpha = 0.85
    x = np.ones(n) / n
    for _ in range(100):
        x = alpha * L @ x + (1 - alpha) / n
    x /= x.sum()
    print(f"  4 页面 PageRank: {np.round(x, 3)}")
    order = np.argsort(-x)
    print(f"  排名顺序 / Ranking: {order}")
    print()
    return x


# =============================================================================
# 演示 5：LU 与 QR 分解
# Demo 5: LU & QR factorizations
# =============================================================================
def demo_lu_qr():
    """LU 用于解线性方程；QR 用于稳定回归。
    LU for solving Ax=b; QR for stable regression.
    """
    print("【演示 5】LU / QR 分解")
    print("[Demo 5] LU / QR factorization")
    print()
    A = np.array([[4, 3], [6, 3]], dtype=float)
    b = np.array([10, 12], dtype=float)
    # numpy 没有内置 LU，用 scipy 或者直接求解
    try:
        from scipy.linalg import lu
        P, L, U = lu(A)
        print(f"  L =\n{L}\n  U =\n{U}")
    except ImportError:
        print("  (scipy 未安装，跳过 LU)")
    Q, R = qr(A)
    print(f"  Q =\n{Q}\n  R =\n{R}")
    x = np.linalg.solve(A, b)
    print(f"  解 Ax=b: x = {x}")
    print()


# =============================================================================
# 演示 6：图像压缩 —— 用前 k 个奇异值
# Demo 6: Image compression — keep top-k singular values
# =============================================================================
def demo_matrix_compression():
    """构造一个 "图像" 矩阵，按 k 截断 SVD，比较误差。
    Build a toy 'image', truncate SVD at k, compare error.
    """
    print("【演示 6】矩阵压缩 (低秩近似)")
    print("[Demo 6] Matrix compression via low-rank approximation")
    print()
    rng = np.random.default_rng(1)
    # 60x60 矩阵，真实秩约为 5
    U0 = rng.standard_normal((60, 5))
    V0 = rng.standard_normal((5, 60))
    M = U0 @ V0 + 0.05 * rng.standard_normal((60, 60))
    U, S, Vt = svd(M, full_matrices=False)
    sizes = []
    errs = []
    for k in [1, 3, 5, 10, 30]:
        Mk = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
        err = np.linalg.norm(M - Mk) / np.linalg.norm(M)
        compress = (60 * k + k + k * 60) / (60 * 60)
        sizes.append(compress)
        errs.append(err)
        print(f"  k={k:>2d}: 相对误差 {err:.4f},  存储占比 {compress:.2%}")
    print()
    return S, sizes, errs


# =============================================================================
# 演示 7：神经网络一层 —— 矩阵乘法 + 激活
# Demo 7: A neural-net layer — matmul + activation
# =============================================================================
def demo_neural_layer():
    """y = ReLU(Wx + b) —— 神经网络的最小积木。
    y = ReLU(Wx + b) — atom of a neural network.
    """
    print("【演示 7】神经网络一层")
    print("[Demo 7] One neural-network layer")
    print()
    rng = np.random.default_rng(7)
    n_in, n_out = 4, 3
    W = rng.standard_normal((n_out, n_in))
    b = rng.standard_normal(n_out)
    x = np.array([1.0, -0.5, 0.3, 2.0])
    z = W @ x + b
    y = np.maximum(0, z)
    print(f"  输入 / input:  {x}")
    print(f"  线性输出 z = Wx + b: {z}")
    print(f"  ReLU 输出 y = max(0, z): {y}")
    print("  ⇒ 一层网络 = 一次矩阵乘法 + 非线性")
    print()


# =============================================================================
# 主函数 / Main
# =============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("linalg_apps.py — 第 8 章演示 / Chapter 8 Demo")
    print("=" * 60)
    print()
    demo_svd()
    X, Vt = demo_pca()
    x, y, coef = demo_least_squares()
    demo_pagerank()
    demo_lu_qr()
    S, sizes, errs = demo_matrix_compression()
    demo_neural_layer()
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
        # PCA 散点
        axes[0].scatter(X[:, 0], X[:, 1], alpha=0.5)
        for i, v in enumerate(Vt):
            axes[0].arrow(X[:, 0].mean(), X[:, 1].mean(),
                          v[0] * 3, v[1] * 3, color=['red', 'green'][i],
                          width=0.05)
        axes[0].set_title('PCA: 数据 + 主轴')
        axes[0].axis('equal')

        # 最小二乘
        axes[1].scatter(x, y, alpha=0.6, label='数据')
        axes[1].plot(x, coef[0] + coef[1] * x, 'r-', label='拟合')
        axes[1].set_title('最小二乘拟合 / Least squares')
        axes[1].legend()

        # 奇异值衰减
        axes[2].semilogy(S, 'o-')
        axes[2].set_xlabel('索引 / index')
        axes[2].set_ylabel('奇异值 / singular value')
        axes[2].set_title('Ch8: SVD 谱衰减')
        axes[2].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig('ch08_linalg.png', dpi=120, bbox_inches='tight')
        plt.close()
        print("\n已保存可视化: ch08_linalg.png")
    except ImportError:
        print("\n(matplotlib 未安装 — 跳过画图)")
