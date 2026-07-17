# -*- coding: utf-8 -*-
"""
问题一+二+三：嵌入式社区养老服务综合优化模型

整合三个模块：
1. 问题一：马尔可夫人口预测与需求分析
2. 问题二：服务站选址与规模优化（分支定界+固定点迭代）
3. 问题三：服务定价与政府补贴优化

运行：python Four.py

依赖：pandas, numpy, openpyxl
"""

import os
import re
import math
import copy
import sys
import io
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# 全局参数配置区

# --- 问题一：马尔可夫预测参数 ---
MARKOV_YEARS = 5                    # 预测年数
MARKOV_MU = 0.05                    # 年死亡率
MARKOV_LAM = 0.07                  # 新增老年人口比例
MARKOV_P12 = 0.045                 # 自理->半失能 转移概率
MARKOV_P23 = 0.10                  # 半失能->失能 转移概率

# --- 问题二：站点选址参数 ---
BUDGET = 120.0                      # 总预算（万元）
SERVICE_RADIUS = 1000              # 服务半径（米）
ALPHA = 0.6                         # 覆盖率权重
BETA = 0.4                          # 满意度权重
MAX_UTIL_ALLOWED = 1.20            # 允许的最大利用率
MAX_ITER = 100                      # 固定点迭代最大次数
TOL = 1e-6                          # 收敛容差
DAMPING = 0.5                       # 阻尼系数

# --- 问题三：定价补贴参数 ---
PROFIT_RATE_UPPER = 0.08            # 利润率上限
SUBSIDY_PER_PERSON_TIME = 2.0      # 人均次补贴（元）
DAYS = 365                          # 年天数
SERVICE_ORDER = ["助餐", "日间照料", "上门护理", "康复理疗", "助浴", "紧急救助"]
NON_EMERGENCY = [s for s in SERVICE_ORDER if s != "紧急救助"]
COMMUNITIES = list("ABCDEFGHIJ")

# 参数文件读取功能

def load_param_file(param_file):
    """读取参数调整文件并更新全局变量"""
    if not param_file.exists():
        print(f"参数文件不存在: {param_file}")
        return False
    
    content = param_file.read_text(encoding="utf-8")
    lines = content.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        # 跳过注释行和空行
        if not line or line.startswith('#'):
            continue
        
        # 解析赋值语句
        if '=' in line:
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()
            
            # 去除行尾注释（# 及其后面的内容）
            if '#' in value:
                value = value[:value.index('#')].strip()
            
            try:
                # 处理Python表达式
                if any(x in key for x in ['MARKOV', 'BUDGET', 'PROFIT', 'SUBSIDY', 'P12', 'P23', 'LAM', 'MU', 'ALPHA', 'BETA', 'MAX', 'TOL', 'DAMPING', 'DAYS', 'DAILY_FIXED_COST_MULTIPLIER']):
                    globals()[key] = float(value)
                else:
                    globals()[key] = eval(value)
            except Exception as e:
                print(f"解析参数 '{key}={value}' 时出错: {e}")
    
    return True

def select_and_load_param():
    """让用户选择要加载的参数文件（支持命令行参数或默认选择）"""
    print("\n" + "=" * 60)
    print("参数文件选择")
    print("=" * 60)
    print("0. 不加载参数文件（使用默认参数）")
    print("1. 加载 参数调整1_人口预测.txt（人口增长率、转移概率）")
    print("2. 加载 参数调整2_成本.txt（日固定管理成本+20%）")
    print("3. 加载 参数调整3_预算.txt（预算调整为140万）")
    print("=" * 60)
    
    # 检查命令行参数
    choice = "0"  # 默认不加载参数文件
    if len(sys.argv) > 1:
        choice = sys.argv[1]
        print(f"命令行参数选择: {choice}")
    else:
        # 在非交互式环境中使用默认选择
        print("未提供命令行参数，使用默认选择 (0)")
    
    if choice == '1':
        print(f"\n正在加载: {PARAM_FILE_1}")
        if load_param_file(PARAM_FILE_1):
            print("参数加载成功！")
            print(f"  MARKOV_LAM = {MARKOV_LAM}")
            print(f"  MARKOV_P12 = {MARKOV_P12}")
            print(f"  MARKOV_P23 = {MARKOV_P23}")
    elif choice == '2':
        print(f"\n正在加载: {PARAM_FILE_2}")
        if load_param_file(PARAM_FILE_2):
            print("参数加载成功！")
            print(f"  日固定管理成本乘数 = 1.2")
    elif choice == '3':
        print(f"\n正在加载: {PARAM_FILE_3}")
        if load_param_file(PARAM_FILE_3):
            print("参数加载成功！")
            print(f"  BUDGET = {BUDGET}")
    else:
        print("\n使用默认参数运行")
    
    print()
    return choice  # 返回选择值用于命名输出文件

# 路径配置

ROOT = Path(__file__).resolve().parent.parent  # 项目根目录（src/ 的上层）
DATA_DIR = ROOT / "data"
ATTACH1 = DATA_DIR / "附件1：小区基础数据.xlsx"
ATTACH2 = DATA_DIR / "附件2：服务需求数据.xlsx"
ATTACH3 = DATA_DIR / "附件3：服务站建设与运营成本.xlsx"

# 参数调整文件路径（必须在 ROOT 之后定义）
PARAM_FILE_1 = ROOT / "参数调整1_人口预测.txt"
PARAM_FILE_2 = ROOT / "参数调整2_成本.txt"
PARAM_FILE_3 = ROOT / "参数调整3_预算.txt"

OUT_DIR = ROOT / "results" / "four_output"
OUT_DIR.mkdir(exist_ok=True)

# 第一部分：马尔可夫人口预测与需求分析

def extract_number(val):
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        match = re.search(r'(\d+(?:\.\d+)?)', val)
        if match:
            num = float(match.group(1))
            if '%' in val:
                num /= 100.0
            return num
    return 0.0

def load_attach1():
    df = pd.read_excel(ATTACH1, sheet_name='人口与老人结构', header=1)
    communities = df.iloc[:, 0].values
    N_self = df.iloc[:, 3].values.astype(float)
    N_semi = df.iloc[:, 4].values.astype(float)
    N_dis = df.iloc[:, 5].values.astype(float)
    income = df.iloc[:, 6].values.astype(float)
    return communities, N_self, N_semi, N_dis, income

def load_attach2():
    df_req = pd.read_excel(ATTACH2, sheet_name='每位老人月均服务需求次数', header=1)
    q_self = df_req.iloc[:, 1].values.astype(float)
    q_semi = df_req.iloc[:, 2].values.astype(float)
    q_dis = df_req.iloc[:, 3].values.astype(float)
    services = df_req.iloc[:, 0].values
    
    df_price = pd.read_excel(ATTACH2, sheet_name='服务营收及支出', header=1)
    revenue_raw = df_price.iloc[:, 1].values
    base_price = np.array([extract_number(v) for v in revenue_raw])
    
    df_limit = pd.read_excel(ATTACH2, sheet_name='月服务消费上限', header=0)
    beta_raw = df_limit.iloc[:, 1].values
    beta = np.array([extract_number(v) for v in beta_raw])
    
    return services, q_self, q_semi, q_dis, base_price, beta

def load_attach3():
    df = pd.read_excel(ATTACH3, sheet_name='服务站建设与运营成本', header=1)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(subset=["站点规模"])
    df = df[df["站点规模"].isin(["小型", "中型", "大型"])].copy()
    
    build_col = [c for c in df.columns if "一次性建设成本" in c][0]
    fixed_col = [c for c in df.columns if "日均固定管理成本" in c][0]
    cap_col = [c for c in df.columns if "日最大服务人次" in c][0]
    
    scale_info = {}
    for _, row in df.iterrows():
        name = row["站点规模"]
        fixed_cost = float(row[fixed_col])
        # 应用成本乘数（如果已加载参数调整2）
        if 'DAILY_FIXED_COST_MULTIPLIER' in globals():
            fixed_cost *= DAILY_FIXED_COST_MULTIPLIER
        scale_info[name] = {
            "build_cost": float(row[build_col]) * 10000,
            "fixed_cost_day": fixed_cost,
            "daily_capacity": float(row[cap_col]),
            "subsidy_limit_day": {"小型": 1000, "中型": 1800, "大型": 2600}.get(name, 0),
        }
    return scale_info

def load_distance_matrix():
    dist = {
        "A": {"A":0,"B":600,"C":1200,"D":900,"E":1500,"F":1800,"G":1300,"H":700,"I":1100,"J":500},
        "B": {"A":600,"B":0,"C":800,"D":500,"E":1100,"F":1400,"G":900,"H":400,"I":700,"J":300},
        "C": {"A":1200,"B":800,"C":0,"D":700,"E":600,"F":900,"G":500,"H":900,"I":600,"J":700},
        "D": {"A":900,"B":500,"C":700,"D":0,"E":800,"F":1100,"G":600,"H":300,"I":500,"J":400},
        "E": {"A":1500,"B":1100,"C":600,"D":800,"E":0,"F":500,"G":400,"H":1000,"I":500,"J":800},
        "F": {"A":1800,"B":1400,"C":900,"D":1100,"E":500,"F":0,"G":500,"H":1200,"I":700,"J":1100},
        "G": {"A":1300,"B":900,"C":500,"D":600,"E":400,"F":500,"G":0,"H":800,"I":400,"J":600},
        "H": {"A":700,"B":400,"C":900,"D":300,"E":1000,"F":1200,"G":800,"H":0,"I":600,"J":300},
        "I": {"A":1100,"B":700,"C":600,"D":500,"E":500,"F":700,"G":400,"H":600,"I":0,"J":400},
        "J": {"A":500,"B":300,"C":700,"D":400,"E":800,"F":1100,"G":600,"H":300,"I":400,"J":0},
    }
    return dist

def markov_population_yearly(s, se, d, years=5, mu=0.05, lam=0.07, p12=0.045, p23=0.10):
    history = [(int(s), int(se), int(d))]
    for _ in range(years):
        total = s + se + d
        s_new = s * (1 - mu) * (1 - p12) + lam * total
        se_new = se * (1 - mu) * (1 - p23) + s * (1 - mu) * p12
        d_new = d * (1 - mu) + se * (1 - mu) * p23
        s, se, d = round(s_new), round(se_new), round(d_new)
        history.append((int(s), int(se), int(d)))
    return history

def run_markov():
    print("=" * 80)
    print("【第一部分】马尔可夫人口预测与需求分析")
    print("=" * 80)
    
    communities, N_self_init, N_semi_init, N_dis_init, income = load_attach1()
    services, q_self, q_semi, q_dis, base_price, beta = load_attach2()
    
    yearly_pop = []
    for i in range(len(communities)):
        hist = markov_population_yearly(
            N_self_init[i], N_semi_init[i], N_dis_init[i],
            years=MARKOV_YEARS, mu=MARKOV_MU, lam=MARKOV_LAM,
            p12=MARKOV_P12, p23=MARKOV_P23
        )
        yearly_pop.append(hist)
    
    N_self_5 = [hist[MARKOV_YEARS][0] for hist in yearly_pop]
    N_semi_5 = [hist[MARKOV_YEARS][1] for hist in yearly_pop]
    N_dis_5 = [hist[MARKOV_YEARS][2] for hist in yearly_pop]
    
    print("\n【1-1】第1年至第5年末各小区各类老人数量")
    print("小区\t年份\t自理\t半失能\t失能\t合计")
    for i, comm in enumerate(communities):
        for t in range(1, MARKOV_YEARS + 1):
            self_t, semi_t, dis_t = yearly_pop[i][t]
            total_t = self_t + semi_t + dis_t
            print(f"{comm}\t第{t}年末\t{self_t}\t{semi_t}\t{dis_t}\t{total_t}")
        print()
    
    print("\n【1-2】第5年末每个小区各项服务的理论月需求（分自理、半失能、失能，未考虑消费约束）")
    print("小区\t老人类型\t服务项目\t理论月需求(次/月)")
    for i, comm in enumerate(communities):
        self_cnt = N_self_5[i]
        semi_cnt = N_semi_5[i]
        dis_cnt = N_dis_5[i]
        for k, (cnt, q_arr) in enumerate([(self_cnt, q_self), (semi_cnt, q_semi), (dis_cnt, q_dis)]):
            type_name = ["自理", "半失能", "失能"][k]
            for m, svc in enumerate(services):
                demand = cnt * q_arr[m]
                demand_int = int(round(demand))
                print(f"{comm}\t{type_name}\t{svc}\t{demand_int}")
    
    base_price_charge = base_price[:5]
    q_self_charge = q_self[:5]
    q_semi_charge = q_semi[:5]
    q_dis_charge = q_dis[:5]
    Q = np.vstack([q_self_charge, q_semi_charge, q_dis_charge])
    Pop5_mat = np.column_stack([N_self_5, N_semi_5, N_dis_5])
    E = np.sum(Q * base_price_charge, axis=1)
    
    alpha = np.ones((len(communities), 3))
    for i, inc in enumerate(income):
        for k in range(3):
            limit = inc * beta[k]
            if E[k] > limit:
                alpha[i, k] = limit / E[k]
    
    constrained_demand = []
    for i in range(len(communities)):
        comm_demand = []
        for k in range(3):
            cnt = Pop5_mat[i, k]
            q_arr = [q_self, q_semi, q_dis][k]
            demand_k = []
            for m in range(6):
                if m < 5:
                    raw = cnt * q_arr[m]
                    cut = raw * alpha[i, k]
                    demand_k.append(int(round(cut)))
                else:
                    raw = cnt * q_arr[m]
                    demand_k.append(int(round(raw)))
            comm_demand.append(demand_k)
        constrained_demand.append(comm_demand)
    
    print("\n【1-3】第5年末每个小区各类老人月均服务需求次数（考虑消费约束，等比例削减取整）")
    print("小区\t老人类型\t服务项目\t约束后月需求(次/月)")
    for i, comm in enumerate(communities):
        for k, type_name in enumerate(["自理", "半失能", "失能"]):
            demand_k = constrained_demand[i][k]
            for m, svc in enumerate(services):
                print(f"{comm}\t{type_name}\t{svc}\t{demand_k[m]}")
    
    pop_df = pd.DataFrame({
        "小区": communities,
        "自理": N_self_5,
        "半失能": N_semi_5,
        "失能": N_dis_5,
    })
    pop_df["合计"] = pop_df["自理"] + pop_df["半失能"] + pop_df["失能"]
    
    demand_rows = []
    for i, comm in enumerate(communities):
        for k, type_name in enumerate(["自理", "半失能", "失能"]):
            for m, svc in enumerate(services):
                demand_rows.append({
                    "小区": comm,
                    "老人类型": type_name,
                    "服务项目": svc,
                    "约束后月需求": constrained_demand[i][k][m],
                })
    demand_df = pd.DataFrame(demand_rows)
    
    income_df = pd.DataFrame({"小区": communities, "人均月收入": income})
    
    return pop_df, demand_df, income_df, base_price

# 第二部分：站点选址与规模优化

ELDER_TYPES = ["自理", "半失能", "失能"]

def get_monthly_demand(demand_df):
    result = {c: 0 for c in COMMUNITIES}
    for _, row in demand_df.iterrows():
        result[row["小区"]] += row["约束后月需求"]
    return result

def get_elderly_by_type(pop_df):
    result = {}
    for _, row in pop_df.iterrows():
        result[row["小区"]] = {
            "自理": int(row["自理"]),
            "半失能": int(row["半失能"]),
            "失能": int(row["失能"]),
        }
    return result

def total_elderly_by_community(elderly_by_type):
    return {c: sum(elderly_by_type[c].values()) for c in COMMUNITIES}

def distance_satisfaction(d):
    if d <= 300: return 1.00
    elif d <= 500: return 0.90
    elif d <= 650: return 0.75
    elif d <= 1000: return 0.60
    else: return 0.0

def response_satisfaction(u):
    if u <= 0.60: return 1.00
    elif u <= 0.75: return 0.93
    elif u <= 0.85: return 0.85
    elif u <= 0.95: return 0.72
    elif u <= 1.00: return 0.60
    else: return 0.30

def total_satisfaction(s1, s2):
    if s1 <= 0: return 0.0
    return 0.2 * s1 + 0.3 * s2 + 0.5

def built_sites(plan):
    return [c for c in COMMUNITIES if plan[c] > 0]

def plan_cost(plan):
    scale_names = {1: "小型", 2: "中型", 3: "大型"}
    total = 0.0
    for c in COMMUNITIES:
        scale = plan[c]
        if scale > 0:
            total += scale_info[scale_names[scale]]["build_cost"] / 10000
    return total

def station_capacity(plan, site):
    scale_names = {1: "小型", 2: "中型", 3: "大型"}
    return scale_info[scale_names[plan[site]]]["daily_capacity"]

def reachable_sites(plan, comm):
    return [s for s in COMMUNITIES if plan[s] > 0 and DIST[comm][s] <= SERVICE_RADIUS]

@dataclass
class AllocationResult:
    feasible: bool
    assignments: Dict[str, Optional[str]]
    station_load: Dict[str, float]
    station_util: Dict[str, float]
    station_s2: Dict[str, float]
    community_sat: Dict[str, float]
    coverage_rate: float
    avg_satisfaction: float
    objective_value: float

@dataclass
class Node:
    idx: int
    plan: Dict[str, int]
    cost: float

@dataclass
class BestSolution:
    plan: Dict[str, int]
    result: AllocationResult
    cost: float

def fixed_point_allocation(plan, monthly_demand, elderly_by_comm):
    sites = built_sites(plan)
    DAYS_PER_MONTH = 30.0

    if not sites:
        return AllocationResult(
            feasible=True,
            assignments={c: None for c in COMMUNITIES},
            station_load={},
            station_util={},
            station_s2={},
            community_sat={c: 0.0 for c in COMMUNITIES},
            coverage_rate=0.0,
            avg_satisfaction=0.0,
            objective_value=0.0
        )

    util = {s: 0.5 for s in sites}
    prev_assignments = None

    for _ in range(MAX_ITER):
        s2 = {s: response_satisfaction(util[s]) for s in sites}
        assignments = {}
        community_sat = {}

        for c in COMMUNITIES:
            cand = reachable_sites(plan, c)
            if not cand:
                assignments[c] = None
                community_sat[c] = 0.0
                continue

            best_site = None
            best_sat = -1.0
            for s in cand:
                s1 = distance_satisfaction(DIST[c][s])
                sat = total_satisfaction(s1, s2[s])
                if sat > best_sat:
                    best_sat = sat
                    best_site = s

            assignments[c] = best_site
            community_sat[c] = best_sat

        station_load = {s: 0.0 for s in sites}
        for c in COMMUNITIES:
            s = assignments[c]
            if s is None: continue
            station_load[s] += monthly_demand[c] * community_sat[c] / DAYS_PER_MONTH

        new_util = {}
        for s in sites:
            cap = station_capacity(plan, s)
            new_util[s] = station_load[s] / cap if cap > 0 else 0.0

        updated_util = {}
        for s in sites:
            updated_util[s] = DAMPING * new_util[s] + (1 - DAMPING) * util[s]

        max_diff = max(abs(updated_util[s] - util[s]) for s in sites)
        same_assign = (assignments == prev_assignments)

        util = updated_util
        prev_assignments = assignments

        if same_assign and max_diff < TOL:
            break

    final_s2 = {s: response_satisfaction(util[s]) for s in sites}
    final_assignments = {}
    final_community_sat = {}

    for c in COMMUNITIES:
        cand = reachable_sites(plan, c)
        if not cand:
            final_assignments[c] = None
            final_community_sat[c] = 0.0
            continue

        best_site = None
        best_sat = -1.0
        for s in cand:
            s1 = distance_satisfaction(DIST[c][s])
            sat = total_satisfaction(s1, final_s2[s])
            if sat > best_sat:
                best_sat = sat
                best_site = s

        final_assignments[c] = best_site
        final_community_sat[c] = best_sat

    final_station_load = {s: 0.0 for s in sites}
    for c in COMMUNITIES:
        s = final_assignments[c]
        if s is None: continue
        final_station_load[s] += monthly_demand[c] * final_community_sat[c] / DAYS_PER_MONTH

    final_station_util = {}
    for s in sites:
        cap = station_capacity(plan, s)
        final_station_util[s] = final_station_load[s] / cap if cap > 0 else 0.0

    for s in sites:
        if final_station_util[s] > MAX_UTIL_ALLOWED + 1e-12:
            return AllocationResult(
                feasible=False,
                assignments=final_assignments,
                station_load=final_station_load,
                station_util=final_station_util,
                station_s2={ss: response_satisfaction(final_station_util[ss]) for ss in sites},
                community_sat=final_community_sat,
                coverage_rate=0.0,
                avg_satisfaction=0.0,
                objective_value=-1.0
            )

    covered_pop = sum(elderly_by_comm[c] for c in COMMUNITIES if final_assignments[c] is not None)
    coverage_rate = covered_pop / sum(elderly_by_comm.values())

    avg_satisfaction = sum(
        elderly_by_comm[c] * final_community_sat[c] for c in COMMUNITIES
    ) / sum(elderly_by_comm.values())

    objective_value = ALPHA * coverage_rate + BETA * avg_satisfaction

    return AllocationResult(
        feasible=True,
        assignments=final_assignments,
        station_load=final_station_load,
        station_util=final_station_util,
        station_s2={s: response_satisfaction(final_station_util[s]) for s in sites},
        community_sat=final_community_sat,
        coverage_rate=coverage_rate,
        avg_satisfaction=avg_satisfaction,
        objective_value=objective_value
    )

def optimistic_upper_bound(node):
    return 1.0

def branch_and_bound(monthly_demand, elderly_by_comm):
    init_plan = {c: 0 for c in COMMUNITIES}

    best = BestSolution(
        plan=copy.deepcopy(init_plan),
        result=AllocationResult(
            feasible=True,
            assignments={c: None for c in COMMUNITIES},
            station_load={},
            station_util={},
            station_s2={},
            community_sat={c: 0.0 for c in COMMUNITIES},
            coverage_rate=0.0,
            avg_satisfaction=0.0,
            objective_value=0.0
        ),
        cost=0.0
    )

    order = sorted(COMMUNITIES, key=lambda c: monthly_demand[c], reverse=True)

    def dfs(node):
        nonlocal best

        if node.cost > BUDGET + 1e-12:
            return

        ub = optimistic_upper_bound(node)
        if ub <= best.result.objective_value + 1e-12:
            return

        if node.idx == len(order):
            result = fixed_point_allocation(node.plan, monthly_demand, elderly_by_comm)
            if not result.feasible:
                return

            if result.objective_value > best.result.objective_value + 1e-12:
                best = BestSolution(
                    plan=copy.deepcopy(node.plan),
                    result=result,
                    cost=node.cost
                )
            return

        comm = order[node.idx]

        for scale in [3, 2, 1, 0]:
            new_plan = copy.deepcopy(node.plan)
            new_plan[comm] = scale
            new_cost = plan_cost(new_plan)

            if new_cost > BUDGET + 1e-12:
                continue

            child = Node(
                idx=node.idx + 1,
                plan=new_plan,
                cost=new_cost
            )
            dfs(child)

    dfs(Node(idx=0, plan=init_plan, cost=0.0))
    return best, order

def run_bnb(demand_df, pop_df):
    print("\n" + "=" * 80)
    print("【第二部分】服务站选址与规模优化")
    print("=" * 80)
    
    monthly_demand = get_monthly_demand(demand_df)
    elderly_by_type = get_elderly_by_type(pop_df)
    elderly_by_comm = total_elderly_by_community(elderly_by_type)
    
    print(f"\n求解中（预算={BUDGET}万元，服务半径={SERVICE_RADIUS}米）...")
    best, order = branch_and_bound(monthly_demand, elderly_by_comm)
    
    scale_names = {0: "不建", 1: "小型", 2: "中型", 3: "大型"}
    
    print("\n【2-1】最优站点方案")
    for c in sorted(COMMUNITIES):
        print(f"站点{c}: {scale_names[best.plan[c]]}")
    
    print(f"\n总建设成本: {best.cost:.2f}万元")
    print(f"覆盖率: {best.result.coverage_rate*100:.2f}%")
    print(f"平均满意度: {best.result.avg_satisfaction:.4f}")
    print(f"目标函数值: {best.result.objective_value:.4f}")
    
    print("\n【2-2】社区分配与满意度")
    print("社区\t服务站\t距离满意度S1\t响应满意度S2\t综合满意度")
    for c in sorted(COMMUNITIES):
        s = best.result.assignments[c]
        if s is None:
            print(f"{c}\t无\t0.00\t0.00\t0.00")
        else:
            s1 = distance_satisfaction(DIST[c][s])
            s2 = best.result.station_s2[s]
            sat = best.result.community_sat[c]
            print(f"{c}\t{s}\t{s1:.2f}\t{s2:.2f}\t{sat:.4f}")
    
    print("\n【2-3】站点负载与利用率")
    print("站点\t规模\t日负载\t容量\t利用率\t响应满意度S2")
    for s in sorted(built_sites(best.plan)):
        size = scale_names[best.plan[s]]
        load = best.result.station_load[s]
        cap = station_capacity(best.plan, s)
        util = best.result.station_util[s]
        s2 = best.result.station_s2[s]
        print(f"{s}\t{size}\t{load:.2f}\t{cap}\t{util:.4f}\t{s2:.2f}")
    
    station_sizes = {c: scale_names[best.plan[c]] for c in COMMUNITIES if best.plan[c] > 0}
    
    assign_rows = []
    for c in COMMUNITIES:
        s = best.result.assignments[c]
        if s:
            assign_rows.append({
                "小区": c,
                "服务站": s,
                "综合满意度_问题二": best.result.community_sat[c],
            })
    assign_df = pd.DataFrame(assign_rows)
    
    load_rows = []
    for s in sorted(built_sites(best.plan)):
        load_rows.append({
            "服务站": s,
            "规模": scale_names[best.plan[s]],
            "日负载": best.result.station_load[s],
            "利用率": best.result.station_util[s],
            "响应满意度S2": best.result.station_s2[s],
        })
    load_df = pd.DataFrame(load_rows)
    
    bnb_summary = {
        "覆盖率": best.result.coverage_rate,
        "平均满意度": best.result.avg_satisfaction,
        "目标函数值": best.result.objective_value,
        "总建设成本_万元": best.cost,
    }
    
    return station_sizes, assign_df, load_df, bnb_summary

# 第三部分：服务定价与政府补贴优化

def to_float(x):
    if pd.isna(x): return np.nan
    if isinstance(x, (int, float, np.number)): return float(x)
    m = re.search(r"-?\d+(?:\.\d+)?", str(x))
    return float(m.group()) if m else np.nan

def price_satisfaction(price, base_price):
    if base_price <= 0:
        return 1.0 if price <= 0 else 0.6
    ratio = price / base_price
    if ratio <= 1.0: return 1.00
    if ratio <= 1.10: return 0.90
    if ratio <= 1.20: return 0.75
    return 0.60

def ceil_to_cent(x):
    return math.ceil((x - 1e-12) * 100) / 100

def load_service_params():
    df = pd.read_excel(ATTACH2, sheet_name="服务营收及支出", header=1)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(subset=["服务项目"])
    df = df[df["服务项目"].isin(SERVICE_ORDER)].copy()
    
    revenue_col = [c for c in df.columns if "营收" in c][0]
    cost_col = [c for c in df.columns if "直接支出" in c][0]
    
    df["基准价格"] = df[revenue_col].apply(to_float).fillna(0.0)
    df["单次直接支出"] = df[cost_col].apply(to_float).fillna(0.0)
    
    base_price = dict(zip(df["服务项目"], df["基准价格"]))
    direct_cost = dict(zip(df["服务项目"], df["单次直接支出"]))
    
    return base_price, direct_cost

def run_gsm(demand_df, pop_df, income_df, station_sizes, assign_df, load_df, bnb_summary, filename_suffix=""):
    """定价与补贴优化（问题三）"""
    print("\n" + "=" * 80)
    print("【第三部分】服务定价与政府补贴优化")
    print("=" * 80)
    
    base_price, direct_cost = load_service_params()
    
    df = demand_df.merge(assign_df, on="小区", how="left")
    
    df["实际有效月需求"] = df["约束后月需求"] * df["综合满意度_问题二"]
    df["实际有效年需求"] = df["实际有效月需求"] * 12
    
    df["基准价格"] = df["服务项目"].map(base_price)
    df["单次直接支出"] = df["服务项目"].map(direct_cost)
    
    df["基准年收入"] = df["实际有效年需求"] * df["基准价格"]
    df["年直接支出"] = df["实际有效年需求"] * df["单次直接支出"]
    
    pricing_rows = []
    profit_rows = []
    
    for site, size in station_sizes.items():
        sub = df[df["服务站"] == site].copy()
        
        base_revenue = sub["基准年收入"].sum()
        direct_total = sub["年直接支出"].sum()
        
        sp = scale_info[size]
        fixed_cost = sp["fixed_cost_day"] * DAYS
        depreciation = sp["build_cost"] / 20.0
        total_cost = direct_total + fixed_cost + depreciation
        
        non_emergency_qty = sub.loc[sub["服务项目"] != "紧急救助", "实际有效年需求"].sum()
        
        subsidy_raw = SUBSIDY_PER_PERSON_TIME * non_emergency_qty
        subsidy_limit = sp["subsidy_limit_day"] * DAYS
        subsidy = min(subsidy_raw, subsidy_limit)
        
        rho = (total_cost - subsidy) / base_revenue if base_revenue > 0 else 0.0
        feasible_under_base = rho <= 1.0
        rho_for_price = min(max(rho, 0.0), 1.0) if feasible_under_base else rho
        
        price_dict = {}
        for service in SERVICE_ORDER:
            if service == "紧急救助":
                price = 0.0
            else:
                price = ceil_to_cent(base_price[service] * rho_for_price)
            price_dict[service] = price
            
            pricing_rows.append({
                "服务站": site,
                "规模": size,
                "折扣系数rho": rho_for_price,
                "服务项目": service,
                "基准价格": base_price[service],
                "最优定价": price,
                "价格满意度S3": price_satisfaction(price, base_price[service]),
                "基准内可保本": feasible_under_base,
            })
        
        sub["最优定价"] = sub["服务项目"].map(price_dict)
        service_revenue = (sub["实际有效年需求"] * sub["最优定价"]).sum()
        
        profit = service_revenue + subsidy - total_cost
        profit_rate = profit / total_cost if total_cost > 0 else np.nan
        
        profit_rows.append({
            "服务站": site,
            "规模": size,
            "基准年收入": base_revenue,
            "优化后年服务收入": service_revenue,
            "年直接支出": direct_total,
            "年固定管理成本": fixed_cost,
            "年折旧成本": depreciation,
            "年度总成本": total_cost,
            "非紧急实际有效年人次": non_emergency_qty,
            "按2元计算补贴": subsidy_raw,
            "年补贴上限": subsidy_limit,
            "实际政府补贴": subsidy,
            "预计年利润": profit,
            "利润率": profit_rate,
            "是否满足利润率≤8%": profit_rate <= PROFIT_RATE_UPPER,
            "是否保本": profit >= -1e-6,
        })
    
    pricing_df = pd.DataFrame(pricing_rows)
    profit_df = pd.DataFrame(profit_rows).sort_values("服务站")
    
    pricing_wide = pricing_df.pivot_table(
        index=["服务站", "规模", "折扣系数rho"],
        columns="服务项目",
        values="最优定价",
        aggfunc="first"
    ).reindex(columns=SERVICE_ORDER).reset_index()
    
    price_s3_map = pricing_df.set_index(["服务站", "服务项目"])["价格满意度S3"].to_dict()
    df["价格满意度S3"] = df.apply(
        lambda r: price_s3_map[(r["服务站"], r["服务项目"])], axis=1
    )
    
    comm_price_sat = df.assign(weight=df["实际有效年需求"]).groupby("小区").apply(
        lambda g: np.average(g["价格满意度S3"], weights=g["weight"]) if g["weight"].sum() > 0 else np.nan,
        include_groups=False
    ).reset_index(name="价格满意度S3")
    
    comm_sat_df = assign_df.merge(comm_price_sat, on="小区", how="left").rename(
        columns={"综合满意度_问题二": "综合满意度"}
    ).sort_values("小区")
    
    price_map = pricing_df.set_index(["服务站", "服务项目"])["最优定价"].to_dict()
    df["优化后年支付"] = df.apply(
        lambda r: r["实际有效年需求"] * price_map[(r["服务站"], r["服务项目"])], axis=1
    )
    
    pop_long = pop_df.melt(
        id_vars="小区",
        value_vars=["自理", "半失能", "失能"],
        var_name="老人类型",
        value_name="人数"
    )
    
    pop_income = pop_long.merge(income_df, on="小区", how="left")
    
    access_rows = []
    for elder_type in ["自理", "半失能", "失能"]:
        n = pop_long.loc[pop_long["老人类型"] == elder_type, "人数"].sum()
        base_month = df.loc[df["老人类型"] == elder_type, "基准年收入"].sum() / 12 / n
        opt_month = df.loc[df["老人类型"] == elder_type, "优化后年支付"].sum() / 12 / n
        
        tmp_income = pop_income[pop_income["老人类型"] == elder_type]
        income_weighted = (tmp_income["人数"] * tmp_income["人均月收入"]).sum() / n
        
        access_rows.append({
            "老人类型": elder_type,
            "第5年末人数": n,
            "加权人均月收入": income_weighted,
            "基准价下人均月支付": base_month,
            "优化后人均月支付": opt_month,
            "支付降幅": 1 - opt_month / base_month if base_month else np.nan,
            "优化后支出占收入": opt_month / income_weighted if income_weighted else np.nan,
        })
    
    access_df = pd.DataFrame(access_rows)
    
    print("\n【3-1】最优定价（元/次）")
    print(pricing_wide.round(4).to_string(index=False))
    
    print("\n【3-2】站点利润与补贴")
    cols = ["服务站", "规模", "优化后年服务收入", "年度总成本", "实际政府补贴", "预计年利润", "利润率"]
    print(profit_df[cols].round(4).to_string(index=False))
    
    print("\n【3-3】社区满意度")
    print(comm_sat_df.round(4).to_string(index=False))
    
    print("\n【3-4】类型可及性")
    print(access_df.round(4).to_string(index=False))
    
    print(f"\n覆盖率: {bnb_summary['覆盖率']*100:.2f}%")
    print(f"平均满意度: {bnb_summary['平均满意度']:.4f}")
    
    OUT_XLSX = OUT_DIR / f"Four_output{filename_suffix}.xlsx"
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        pop_df.to_excel(writer, sheet_name="人口预测", index=False)
        demand_df.to_excel(writer, sheet_name="需求数据", index=False)
        
        if not assign_df.empty:
            assign_df.to_excel(writer, sheet_name="社区分配", index=False)
        if not load_df.empty:
            load_df.to_excel(writer, sheet_name="站点负载", index=False)
        
        pricing_df.to_excel(writer, sheet_name="定价明细", index=False)
        pricing_wide.to_excel(writer, sheet_name="定价汇总", index=False)
        profit_df.to_excel(writer, sheet_name="利润补贴", index=False)
        comm_sat_df.to_excel(writer, sheet_name="社区满意度", index=False)
        access_df.to_excel(writer, sheet_name="类型可及性", index=False)
    
    return pricing_df, profit_df, comm_sat_df, access_df

# =========================================================
# 主函数
# =========================================================

def main():
    output_buffer = []
    
    def capture_print(*args, sep=' ', end='\n', file=None, flush=False):
        output_buffer.append(sep.join(str(arg) for arg in args) + end)
    
    old_print = print
    __builtins__.print = capture_print
    
    try:
        # 选择并加载参数文件，获取选择值
        choice = select_and_load_param()
        
        # 根据选择生成文件名后缀
        suffix_map = {
            "0": "_默认参数",
            "1": "_人口预测调整",
            "2": "_成本调整",
            "3": "_预算调整"
        }
        suffix = suffix_map.get(choice, "_未知")
        
        global scale_info, DIST
        scale_info = load_attach3()
        DIST = load_distance_matrix()
        
        pop_df, demand_df, income_df, base_price = run_markov()
        station_sizes, assign_df, load_df, bnb_summary = run_bnb(demand_df, pop_df)
        pricing_df, profit_df, comm_sat_df, access_df = run_gsm(
            demand_df, pop_df, income_df, station_sizes, assign_df, load_df, bnb_summary,
            filename_suffix=suffix
        )
        
        print("\n" + "=" * 80)
        print("【执行完成】所有结果已保存")
        print("=" * 80)
        
    finally:
        __builtins__.print = old_print
        output_content = ''.join(output_buffer)
        
        # 根据参数选择生成不同的输出文件名
        txt_filename = f"Four_output{suffix}.txt"
        xlsx_filename = f"Four_output{suffix}.xlsx"
        
        OUT_TXT = OUT_DIR / txt_filename
        OUT_TXT.write_text(output_content, encoding="utf-8-sig")
        print(f"输出已保存：{OUT_TXT}")
        print(f"Excel已保存：{OUT_DIR / xlsx_filename}")

if __name__ == "__main__":
    main()
