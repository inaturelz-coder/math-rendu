# 打通工科数学任督二脉 / Math Rendu

> **Bridging the Threads of Engineering Mathematics**
> *从同济高数到 ChatGPT 训练——工科数学的贯通课*

Author: **Li Zhou**
Email: lizhou_alfred2011@hotmail.com
第一版 / First Edition: 2027
许可证 / License: MIT

---

## 关于本书 / About

这是《打通系列》的第二弹——继《打通物理任督二脉》之后，写给**工科学生、跨学科研究生、数学背景薄弱的工程师**的一本"贯通式"数学讲义。

它不是同济高数的替代品，不是 Apostol/Rudin 的中文版，更不是网红科普。它是一本——**用工程师的语言，把微积分到优化的 13 个核心数学话题讲穿、讲透、讲到能写代码**——的书。

**核心特色 / Features**

- **13 章必修 + 1 章可选**：微积分 → 矢量 → ODE/PDE → 线代（3 章重点）→ 复变 → 概率统计 → 数值 → 优化
- **6 部分结构**：分析主线（Ch 1-5, 9）+ 代数主线（Ch 6-8）+ 应用主线（Ch 10-13）
- **每章 7 件套**：本质 / 教材 / 直觉 / 数学 / 例子 / 现代 / 代码
- **每章小故事盒**：Newton 与瘟疫之年、Fourier 被拒稿三次、Larry Page 的博士论文……数学不是凭空发明，是某些人在某个时代解决某个问题
- **Aha 例子工业级**：搜索引擎 / 推荐系统 / GPS 定位 / ChatGPT 训练——不是教科书的玩具问题
- **13 个 Python 模块**：MIT 开源，每章 6+ 数值实验
- **双语**：中文为主，关键术语配英文

---

## 仓库结构 / Repository Structure

```
math-rendu/
├── README.md                  ← 本文件
├── LICENSE                    ← MIT 许可证
├── requirements.txt           ← Python 依赖
├── book.tex                   ← 主 LaTeX 源文件
├── chapters/                  ← 各章源文件 + 单章 PDF
│   ├── ch00_preface.tex
│   ├── ch01_calculus_intro.tex
│   ├── ch02_multi_calc.tex
│   ├── ch03_vector_analysis.tex
│   ├── ch04_ode.tex
│   ├── ch05_pde.tex
│   ├── ch06_linalg_intro.tex
│   ├── ch07_linalg_core.tex
│   ├── ch08_linalg_apps.tex
│   ├── ch09_complex_analysis.tex
│   ├── ch10_probability.tex
│   ├── ch11_statistics.tex
│   ├── ch12_numerical.tex
│   ├── ch13_optimization.tex
│   └── ch14_discrete.tex      (可选)
├── modules/                   ← 13 + 1 个配套 Python 模块
│   ├── calculus_intro.py      Ch 1 微积分
│   ├── multi_calc.py          Ch 2 多元微积分
│   ├── vector_analysis.py     Ch 3 矢量分析
│   ├── ode.py                 Ch 4 ODE
│   ├── pde.py                 Ch 5 PDE
│   ├── linalg_intro.py        Ch 6 线代入门
│   ├── linalg_core.py         Ch 7 线代核心
│   ├── linalg_apps.py         Ch 8 线代应用
│   ├── complex_analysis.py    Ch 9 复变
│   ├── probability.py         Ch 10 概率
│   ├── statistics.py          Ch 11 统计
│   ├── numerical.py           Ch 12 数值
│   ├── optimization.py        Ch 13 优化
│   └── discrete.py            Ch 14 离散 (可选)
├── docs/                      ← 完整书 PDF
│   └── math_rendu_v1.pdf
└── scripts/                   ← 辅助脚本
    └── test_all_modules.py
```

---

## 风格定调 / Style Manifesto

参考：吴军《数学之美》（中信出版社）

- **从工程问题切入**——不是"概念 → 应用"，而是"问题 → 数学"
- **公式量节制**——必要才出现，不堆叠
- **故事 + 人物 + 历史**——数学不是凭空发明
- **接地气**——中国工程师的语言习惯
- **克制的优雅**——不浮夸、不嬉笑
- **应用导向**——Google / 微软 / AI 公司的真实场景

气质参照：

- 像吴军写贾里尼克那样——有人物、有时代
- 像 Strogatz 但更克制——数学之美但不撒狗血
- 像一位老工程师在咖啡馆给后辈讲数学——严肃但不严厉

---

## 如何使用 / How to Use

**完整阅读**（4-6 周）：Ch 0 → Ch 1-13，按顺序

**速成路径**（2 周）：

- 工程师补线代：Ch 6-8
- 工程师补 ML 数学：Ch 6-8 + Ch 10-11 + Ch 13
- 物理学家补现代数学：Ch 8 + Ch 10 + Ch 12-13
- 程序员补理论数学：Ch 1, Ch 6-8, Ch 10, Ch 13

**代码并行**：每章 §7 给出 Python 模块的 5-6 个 demo——边读边跑。

---

## 与《打通物理任督二脉》的关系

- **品牌呼应**："打通系列"第二弹
- **风格继承**：7 件套结构 + 数值实验 + 双语 + GitHub 开源
- **内容独立**：数学为主体，物理只作为应用之一
- **交叉引用**：本书 Ch 8 量子比特 ↔ 物理版 Ch 17 量子计算；本书 Ch 13 优化 ↔ 物理版 Ch 21 物理与 AI

---

## 长期规划 / Roadmap

- **2026**：《打通物理任督二脉》 — 22 章 / 480 页 ✅
- **2027 上**：《打通工科数学任督二脉》 — 13 章 / 280 页 ← **本书**
- **2027 下**：《打通科学计算任督二脉》或《打通材料基础任督二脉》
- **2028**：《打通 CALPHAD 任督二脉》 / 武侠版

---

文责自负。

---

**GitHub**: https://github.com/inaturelz-coder/math-rendu

**Email**: lizhou_alfred2011@hotmail.com

数学是活的——但它需要有人来贯通。
