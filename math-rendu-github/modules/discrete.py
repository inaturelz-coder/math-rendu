"""
discrete.py — 离散数学
========================
Module 14 (optional) of "Math is Alive" / 《数学是活的》第 14 章

实现 / Implements:
    - 排列 / 组合
    - 图的表示 (邻接表 / 矩阵)
    - BFS / DFS
    - Dijkstra 最短路径
    - 旅行商问题 (TSP) 暴力解
    - 复杂度对比 (O(n) vs O(n²) vs O(2ⁿ))

数学基础 / Math:
    C(n,k) = n! / (k!(n-k)!)
    Dijkstra: 贪心 + 优先队列
    TSP: NP-hard

Li Zhou
2027 / MIT License
"""

import numpy as np
from itertools import permutations, combinations
from collections import deque
from math import factorial
import heapq
import time


# =============================================================================
# 演示 1：排列 / 组合
# Demo 1: Permutations & combinations
# =============================================================================
def demo_combinations():
    """C(n, k) 与 P(n, k) 的计数与列举。
    Count and enumerate permutations and combinations.
    """
    print("【演示 1】排列与组合")
    print("[Demo 1] Permutations & combinations")
    print()
    n, k = 5, 3
    P = factorial(n) // factorial(n - k)
    C = factorial(n) // (factorial(k) * factorial(n - k))
    print(f"  P(n={n}, k={k}) = {P}")
    print(f"  C(n={n}, k={k}) = {C}")
    items = list("ABCDE")
    perms = list(permutations(items, k))
    combs = list(combinations(items, k))
    print(f"  列举前 3 个排列: {perms[:3]}")
    print(f"  列举前 3 个组合: {combs[:3]}")
    print(f"  长度匹配: P={len(perms)}, C={len(combs)}")
    print()


# =============================================================================
# 演示 2：图基础
# Demo 2: Graph basics
# =============================================================================
def demo_graph_basics():
    """邻接表 + 邻接矩阵。
    Adjacency list + matrix representations.
    """
    print("【演示 2】图的基本表示")
    print("[Demo 2] Graph representations")
    print()
    # 5 个节点的图
    edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4)]
    n = 5
    adj_list = {i: [] for i in range(n)}
    adj_mat = np.zeros((n, n), dtype=int)
    for u, v in edges:
        adj_list[u].append(v)
        adj_list[v].append(u)
        adj_mat[u, v] = adj_mat[v, u] = 1
    print(f"  顶点 = {n}, 边 = {len(edges)}")
    print(f"  邻接表 / Adj list:")
    for k, v in adj_list.items():
        print(f"    {k}: {v}")
    print(f"  邻接矩阵 /Adj matrix:\n{adj_mat}")
    print()
    return adj_list


# =============================================================================
# 演示 3：BFS / DFS
# Demo 3: BFS / DFS
# =============================================================================
def demo_bfs_dfs(adj_list=None):
    """从节点 0 出发，对比 BFS 与 DFS 的访问顺序。
    Starting at node 0, compare BFS and DFS visit orders.
    """
    print("【演示 3】BFS 与 DFS")
    print("[Demo 3] BFS & DFS traversals")
    print()
    if adj_list is None:
        adj_list = {0: [1, 2], 1: [0, 2, 3], 2: [0, 1, 3], 3: [1, 2, 4],
                    4: [3]}

    def bfs(start):
        visited, order = {start}, []
        q = deque([start])
        while q:
            u = q.popleft()
            order.append(u)
            for v in adj_list[u]:
                if v not in visited:
                    visited.add(v)
                    q.append(v)
        return order

    def dfs(start):
        visited, order = set(), []

        def go(u):
            visited.add(u)
            order.append(u)
            for v in adj_list[u]:
                if v not in visited:
                    go(v)
        go(start)
        return order

    print(f"  BFS 顺序: {bfs(0)}")
    print(f"  DFS 顺序: {dfs(0)}")
    print()


# =============================================================================
# 演示 4：Dijkstra 最短路径
# Demo 4: Dijkstra's shortest path
# =============================================================================
def demo_dijkstra():
    """加权图上单源最短路径。
    Single-source shortest path on a weighted graph.
    """
    print("【演示 4】Dijkstra 最短路径")
    print("[Demo 4] Dijkstra shortest path")
    print()
    # (u, v, w)
    edges = [(0, 1, 4), (0, 2, 1), (2, 1, 2), (1, 3, 1),
             (2, 3, 5), (3, 4, 3)]
    n = 5
    graph = {i: [] for i in range(n)}
    for u, v, w in edges:
        graph[u].append((v, w))
        graph[v].append((u, w))

    def dijkstra(src):
        dist = {i: float('inf') for i in range(n)}
        dist[src] = 0
        pq = [(0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            for v, w in graph[u]:
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        return dist

    d = dijkstra(0)
    print(f"  从节点 0 出发的最短距离:")
    for k, v in d.items():
        print(f"    到 {k}: {v}")
    print()


# =============================================================================
# 演示 5：TSP 暴力解
# Demo 5: TSP brute force
# =============================================================================
def demo_tsp_brute():
    """枚举所有路径，找出最短的环。
    Enumerate all routes, find the shortest cycle.
    """
    print("【演示 5】旅行商问题 (TSP) 暴力解")
    print("[Demo 5] TSP via brute force")
    print()
    rng = np.random.default_rng(0)
    n = 7  # 7 个城市，6! = 720 路线
    coords = rng.uniform(0, 10, size=(n, 2))
    D = np.linalg.norm(coords[:, None] - coords[None], axis=2)

    best_len = float('inf')
    best_route = None
    # 固定起点 0，枚举其余 (n-1)!
    for perm in permutations(range(1, n)):
        route = (0,) + perm + (0,)
        length = sum(D[route[i], route[i + 1]] for i in range(n))
        if length < best_len:
            best_len = length
            best_route = route
    print(f"  城市数 n = {n}, 路径数 = {factorial(n - 1)}")
    print(f"  最短回路长度 = {best_len:.3f}")
    print(f"  路径 = {best_route}")
    print("  ⇒ 随 n 增长爆炸：n=20 时 19! ≈ 10¹⁷ 条路径")
    print()
    return coords, best_route


# =============================================================================
# 演示 6：复杂度对比
# Demo 6: Complexity comparison
# =============================================================================
def demo_complexity_comparison():
    """O(n) vs O(n²) vs O(2ⁿ) 的实测运行时间。
    Empirical runtime: O(n) vs O(n²) vs O(2ⁿ).
    """
    print("【演示 6】复杂度实测")
    print("[Demo 6] Empirical complexity")
    print()
    rng = np.random.default_rng(0)
    sizes = [10, 100, 1000, 5000]
    for n in sizes:
        a = rng.standard_normal(n)
        # O(n) 求和
        t0 = time.perf_counter()
        _ = a.sum()
        t_lin = time.perf_counter() - t0
        # O(n²) 双重循环（用 numpy 模拟）
        t0 = time.perf_counter()
        if n <= 1000:
            _ = (a[:, None] + a[None]).sum()
            t_quad = time.perf_counter() - t0
            t_quad_str = f"{t_quad*1e6:>8.1f} µs"
        else:
            t_quad_str = "      跳过 (太慢)"
        print(f"  n = {n:>4d}: O(n) = {t_lin*1e6:>8.2f} µs,  "
              f"O(n²) = {t_quad_str}")
    # 2ⁿ
    print("  O(2ⁿ): n=20 → 10⁶ 步,  n=40 → 10¹² 步 (一台桌面机算不完)")
    print()


# =============================================================================
# 主函数 / Main
# =============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("discrete.py — 第 14 章演示 / Chapter 14 Demo")
    print("=" * 60)
    print()
    demo_combinations()
    adj_list = demo_graph_basics()
    demo_bfs_dfs(adj_list)
    demo_dijkstra()
    coords, route = demo_tsp_brute()
    demo_complexity_comparison()
    print("=" * 60)
    print("演示完毕 / Demo complete.")
    print("=" * 60)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        # TSP 路径
        axes[0].scatter(coords[:, 0], coords[:, 1], s=80, c='red',
                        zorder=3)
        for i, (xc, yc) in enumerate(coords):
            axes[0].annotate(str(i), (xc, yc),
                             textcoords="offset points", xytext=(6, 6))
        for i in range(len(route) - 1):
            a, b = route[i], route[i + 1]
            axes[0].plot([coords[a, 0], coords[b, 0]],
                         [coords[a, 1], coords[b, 1]], 'b-', alpha=0.6)
        axes[0].set_title("Ch14: TSP 最优回路")
        axes[0].set_aspect('equal')
        axes[0].grid(alpha=0.3)

        # 复杂度曲线
        n = np.arange(1, 21)
        axes[1].semilogy(n, n, label='O(n)')
        axes[1].semilogy(n, n ** 2, label='O(n²)')
        axes[1].semilogy(n, 2.0 ** n, label='O(2ⁿ)')
        axes[1].semilogy(n, np.array([factorial(k) for k in n], dtype=float),
                         label='O(n!)')
        axes[1].set_xlabel('n')
        axes[1].set_ylabel('操作数 (log)')
        axes[1].set_title('复杂度增长 / Complexity growth')
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig('ch14_discrete.png', dpi=120, bbox_inches='tight')
        plt.close()
        print("\n已保存可视化: ch14_discrete.png")
    except ImportError:
        print("\n(matplotlib 未安装 — 跳过画图)")
