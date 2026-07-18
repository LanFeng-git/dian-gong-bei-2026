"""
电工杯 B题 — 敏感性分析可视化
生成图表：多场景对比、人口趋势、选址分布、定价分析
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# 中文字体设置
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans SC']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "viz")
os.makedirs(OUTPUT, exist_ok=True)

# ============================================================
# 数据：提取自 four_output 的 4 组场景（默认/人口/成本/预算）
# ============================================================
scenarios = ["默认参数", "人口预测调整", "成本调整(+20%)", "预算调整(140万)"]

# 4个场景的站点选址结果（A-J 共10个候选站点）
# 数据来源：four_output 各场景 BnB 输出
site_selection = {
    "默认参数":   {"A": 1, "C": 1, "D": 1, "E": 1, "H": 1},
    "人口预测调整": {"A": 1, "C": 1, "D": 1, "E": 1, "F": 1},
    "成本调整(+20%)": {"A": 1, "C": 1, "D": 1, "E": 1},
    "预算调整(140万)": {"A": 1, "C": 1, "D": 1, "E": 1, "H": 1, "I": 1},
}

# 4个场景的关键指标
metrics = pd.DataFrame({
    "场景": scenarios,
    "建站数量": [5, 5, 4, 6],
    "覆盖率(%)": [78.5, 79.2, 68.3, 85.7],
    "综合满意度": [0.87, 0.85, 0.82, 0.88],
    "年成本(万元)": [118.5, 116.2, 137.6, 146.3],
    "预算利用率(%)": [98.8, 96.8, 91.7, 95.2],
})

# 人口预测数据（默认参数，A-J各小区5年趋势）
pop_data = pd.DataFrame({
    "小区": ["A","B","C","D","E","F","G","H","I","J"],
    "第1年末": [726,621,939,555,864,520,953,627,812,726],
    "第3年末": [755,646,978,577,902,538,989,650,842,755],
    "第5年末": [785,672,1018,600,940,556,1026,674,873,785],
})

# ============================================================
# 图1：4场景对比雷达图
# ============================================================
def plot_scenario_radar():
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    categories = ["建站数量", "覆盖率(%)", "综合满意度", "年成本(万元)", "预算利用率(%)"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    colors = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12"]
    for i, row in metrics.iterrows():
        values = row[1:].tolist()
        # 归一化到 0-100 尺度
        max_vals = [6, 100, 1.0, 200, 100]
        norm = [v / m * 100 for v, m in zip(values, max_vals)]
        norm += norm[:1]
        ax.fill(angles, norm, alpha=0.1, color=colors[i])
        ax.plot(angles, norm, 'o-', label=scenarios[i], color=colors[i], linewidth=2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_title("多场景敏感性分析对比", fontsize=16, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0), fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT, "01_多场景敏感性分析对比.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("  + 雷达图")

# ============================================================
# 图2：选址方案对比（柱状图）
# ============================================================
def plot_site_comparison():
    all_sites = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    n_sites = len(all_sites)
    n_scenarios = len(scenarios)

    fig, axes = plt.subplots(1, n_scenarios, figsize=(16, 4), sharey=True)
    for idx, (scenario, sites) in enumerate(site_selection.items()):
        ax = axes[idx]
        selected = [1 if s in sites else 0 for s in all_sites]
        colors = ["#27ae60" if x else "#bdc3c7" for x in selected]
        ax.bar(all_sites, selected, color=colors, width=0.6, edgecolor="white")
        ax.set_title(scenario, fontsize=11)
        ax.set_ylim(0, 1.5)
        ax.set_yticks([0, 1])
        ax.set_xticks(range(n_sites))
        ax.set_xticklabels(all_sites)
        ax.text(0.5, 0.95, f"n={sum(selected)}", transform=ax.transAxes,
                fontsize=10, ha="center", va="top")
    fig.suptitle("各场景选址方案对比", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT, "02_选址方案对比.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("  + 选址方案对比图")

# ============================================================
# 图3：人口预测趋势（折线图）
# ============================================================
def plot_population_trend():
    fig, ax = plt.subplots(figsize=(10, 5))
    x = [1, 3, 5]
    for i, row in pop_data.iterrows():
        ax.plot(x, [row["第1年末"], row["第3年末"], row["第5年末"]],
                "o-", label=f"小区 {row['小区']}", linewidth=1.5, markersize=4)
    ax.set_xlabel("年份", fontsize=12)
    ax.set_ylabel("老年人口(人)", fontsize=12)
    ax.set_title("各小区老年人口趋势预测（马尔可夫链）", fontsize=14)
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT, "03_人口预测趋势.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("  + 人口预测趋势图")

# ============================================================
# 图4：关键指标对比柱状图
# ============================================================
def plot_metrics_bar():
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    bar_colors = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12"]

    metric_names = ["建站数量", "覆盖率(%)", "综合满意度", "预算利用率(%)"]
    metric_keys = ["建站数量", "覆盖率(%)", "综合满意度", "预算利用率(%)"]
    for i, (name, key) in enumerate(zip(metric_names, metric_keys)):
        ax = axes[i]
        vals = metrics[key].values
        bars = ax.bar(scenarios, vals, color=bar_colors, width=0.5)
        ax.set_title(name, fontsize=13)
        ax.tick_params(axis="x", rotation=15, labelsize=9)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("四场景关键指标对比", fontsize=15)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT, "04_关键指标对比.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("  + 关键指标对比图")

# ============================================================
# 运行全部
# ============================================================
if __name__ == "__main__":
    print("生成电工杯可视化图表...")
    plot_scenario_radar()
    plot_site_comparison()
    plot_population_trend()
    plot_metrics_bar()
    print(f"完成! 输出目录: {OUTPUT}")
    for f in sorted(os.listdir(OUTPUT)):
        fpath = os.path.join(OUTPUT, f)
        size = os.path.getsize(fpath)
        print(f"  {f} ({size/1024:.0f} KB)")
