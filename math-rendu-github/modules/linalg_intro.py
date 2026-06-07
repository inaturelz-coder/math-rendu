"""
linalg_intro.py — 线性代数入门模块
==================================
Module 6 of "Math is Alive" / 《数学是活的》第 6 模块

实现 / Implements:
    - 矩阵即线性变换（图形演示）/ Matrix as linear transformation
    - 二维旋转矩阵 / 2D rotation matrix
    - 行列式 = 面积/体积缩放因子 / Determinant as area scaling
    - 矩阵求逆 / Matrix inverse
    - Gauss 消元法 / Gaussian elimination
    - 三维旋转（Rodrigues 公式）/ 3D rotation

Li Zhou
2026 年 6 月 / June 2026
MIT License
"""

import numpy as np


# =============================================================================
# 演示 1：矩阵作为变换 / Matrix as Transformation
# =============================================================================
def demo_matrix_as_transform():
    """把单位方块 [(0,0),(1,0),(1,1),(0,1)] 用 A 作用，观察变形。
    Apply A to unit square corners and watch it morph.
    """
    print("\n[Demo 1] 矩阵即线性变换 / Matrix as transformation")
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]).T
    transforms = {
        '缩放 / Scale':   np.array([[2, 0], [0, 0.5]]),
        '剪切 / Shear':   np.array([[1, 1], [0, 1]]),
        '旋转 45° / Rot': np.array([[np.cos(np.pi/4), -np.sin(np.pi/4)],
                                    [np.sin(np.pi/4),  np.cos(np.pi/4)]]),
        '反射 / Reflect': np.array([[1, 0], [0, -1]]),
    }
    for name, A in transforms.items():
        transformed = A @ square
        print(f"  {name:20s}  顶点 (1,1) → ({transformed[0,2]:+.3f}, "
              f"{transformed[1,2]:+.3f})  det={np.linalg.det(A):+.3f}")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
        fig, axes = plt.subplots(1, len(transforms), figsize=(14, 4))
        for ax, (name, A) in zip(axes, transforms.items()):
            ax.plot(square[0], square[1], 'b-', alpha=0.5, label='原')
            t = A @ square
            ax.plot(t[0], t[1], 'r-', label='变换后')
            ax.set_title(name, fontsize=10); ax.axis('equal')
            ax.grid(alpha=0.3); ax.set_xlim(-2, 3); ax.set_ylim(-2, 3)
        plt.suptitle('Ch6: 矩阵线性变换 / Matrix transformations')
        plt.tight_layout(); plt.savefig('ch06_transform.png', dpi=120); plt.close()
        print("  ✓ 已保存 ch06_transform.png")
    except ImportError:
        pass


# =============================================================================
# 演示 2：二维旋转 / 2D Rotation
# =============================================================================
def demo_rotation_2d():
    """R(θ) = [[cos -sin],[sin cos]]，验证 R(α)·R(β) = R(α+β)。
    Verify rotation composition.
    """
    print("\n[Demo 2] 二维旋转 / 2D rotation composition")
    R = lambda t: np.array([[np.cos(t), -np.sin(t)],
                            [np.sin(t),  np.cos(t)]])
    a, b = np.pi/6, np.pi/4
    lhs = R(a) @ R(b)
    rhs = R(a + b)
    print(f"  ||R(α)R(β) - R(α+β)||  = {np.linalg.norm(lhs - rhs):.3e}")
    print(f"  det R(π/3) = {np.linalg.det(R(np.pi/3)):.6f}  (应 = 1)")


# =============================================================================
# 演示 3：行列式与面积 / Determinant ↔ Area
# =============================================================================
def demo_determinant_area():
    """|det A| = 单位面积被 A 变换后的新面积。
    |det A| = scale factor of area.
    """
    print("\n[Demo 3] 行列式 = 面积比 / Determinant = area scaling")
    for label, A in [
        ('单位矩阵', np.eye(2)),
        ('2 倍放大', 2*np.eye(2)),
        ('剪切',     np.array([[1, 2], [0, 1]])),
        ('反射',     np.array([[1, 0], [0, -1]])),
        ('退化',     np.array([[1, 2], [2, 4]])),
    ]:
        d = np.linalg.det(A)
        print(f"  {label:10s}  det = {d:+.3f}  →  面积 ×|{d:.3f}|")


# =============================================================================
# 演示 4：矩阵求逆 / Matrix Inverse
# =============================================================================
def demo_inverse():
    """A^{-1} 存在当且仅当 det A ≠ 0；A·A^{-1} = I。
    Matrix inverse and verification.
    """
    print("\n[Demo 4] 矩阵求逆 / Matrix inverse")
    A = np.array([[4, 7], [2, 6]], dtype=float)
    A_inv = np.linalg.inv(A)
    print(f"  A    =\n{A}")
    print(f"  A⁻¹  =\n{A_inv}")
    print(f"  A·A⁻¹ =\n{A @ A_inv}")
    print(f"  误差 / Error from I = {np.linalg.norm(A @ A_inv - np.eye(2)):.3e}")


# =============================================================================
# 演示 5：Gauss 消元 / Gaussian Elimination
# =============================================================================
def demo_gauss_elimination():
    """手写 Gauss 消元解 Ax = b，与 np.linalg.solve 对照。
    Hand-written elimination compared with numpy.
    """
    print("\n[Demo 5] Gauss 消元法 / Gaussian elimination")
    A = np.array([[2., 1., -1.],
                  [-3., -1., 2.],
                  [-2., 1., 2.]])
    b = np.array([8., -11., -3.])
    # 增广矩阵
    M = np.hstack([A, b.reshape(-1, 1)])
    n = len(b)
    for i in range(n):
        # 部分主元 / partial pivoting
        max_row = i + np.argmax(np.abs(M[i:, i]))
        if max_row != i:
            M[[i, max_row]] = M[[max_row, i]]
        # 消元 / elimination
        for j in range(i+1, n):
            M[j] -= (M[j, i]/M[i, i]) * M[i]
    # 回代 / back substitution
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (M[i, -1] - M[i, i+1:n] @ x[i+1:n]) / M[i, i]
    x_ref = np.linalg.solve(A, b)
    print(f"  Gauss 解: {x}")
    print(f"  numpy 解: {x_ref}")
    print(f"  误差 = {np.linalg.norm(x - x_ref):.3e}")


# =============================================================================
# 演示 6：三维旋转 / 3D Rotation (Rodrigues)
# =============================================================================
def demo_3d_rotation():
    """Rodrigues 公式：绕单位轴 k 旋转角 θ。
    Rotate vector around axis using Rodrigues' formula.
    """
    print("\n[Demo 6] 三维旋转 / 3D rotation (Rodrigues)")
    def rotation_matrix(axis, theta):
        k = np.asarray(axis, float)
        k = k / np.linalg.norm(k)
        K = np.array([[0, -k[2], k[1]],
                      [k[2], 0, -k[0]],
                      [-k[1], k[0], 0]])
        return np.eye(3) + np.sin(theta)*K + (1 - np.cos(theta))*(K @ K)

    R = rotation_matrix([0, 0, 1], np.pi/2)  # 绕 z 转 90°
    v = np.array([1, 0, 0])
    print(f"  R(z, 90°)·(1,0,0) = {R @ v}  (期望 (0,1,0))")
    print(f"  det R = {np.linalg.det(R):.6f}  (应 = 1)")
    print(f"  R·Rᵀ = I ?  误差 = {np.linalg.norm(R @ R.T - np.eye(3)):.3e}")


# =============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("linalg_intro.py — 数学是活的 / 第 6 模块演示")
    print("=" * 60)
    demo_matrix_as_transform()
    demo_rotation_2d()
    demo_determinant_area()
    demo_inverse()
    demo_gauss_elimination()
    demo_3d_rotation()
    print("\n" + "=" * 60)
    print("演示完毕 / Demo complete.")
    print("=" * 60)
