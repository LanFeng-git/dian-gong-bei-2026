# -*- coding: utf-8 -*-
"""
问题二：嵌入式社区养老服务站选址与规模优化
求解算法：分支定界 + 固定点迭代
采用方案2：
1. 当利用率 u > 1.00 时，响应满意度 S2 = 0.30
2. 允许轻度超载，但要求 u <= 1.20
3. 输出每个服务站的预计年度利润
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import copy
import math

# 基础数据

COMMUNITIES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
SERVICE_TYPES = ["助餐", "日间照料", "上门护理", "康复理疗", "助浴", "紧急救助"]
ELDER_TYPES = ["自理", "半失能", "失能"]

# 附件1：小区基础数据
BASE_DATA = {
    "A": {"total_pop": 3200, "elderly": 712, "self": 496, "semi": 152, "disabled": 64, "income": 3400},
    "B": {"total_pop": 2800, "elderly": 608, "self": 408, "semi": 136, "disabled": 64, "income": 3100},
    "C": {"total_pop": 4100, "elderly": 920, "self": 632, "semi": 208, "disabled": 80, "income": 3800},
    "D": {"total_pop": 2500, "elderly": 544, "self": 368, "semi": 120, "disabled": 56, "income": 2900},
    "E": {"total_pop": 3600, "elderly": 784, "self": 536, "semi": 176, "disabled": 72, "income": 3500},
    "F": {"total_pop": 2200, "elderly": 472, "self": 328, "semi": 104, "disabled": 40, "income": 2700},
    "G": {"total_pop": 3900, "elderly": 864, "self": 592, "semi": 192, "disabled": 80, "income": 3600},
    "H": {"total_pop": 2600, "elderly": 568, "self": 392, "semi": 128, "disabled": 48, "income": 3000},
    "I": {"total_pop": 3400, "elderly": 736, "self": 504, "semi": 168, "disabled": 64, "income": 3300},
    "J": {"total_pop": 3000, "elderly": 656, "self": 456, "semi": 144, "disabled": 56, "income": 3200},
}

# 老人状态转移参数
P_SELF_TO_SEMI = 0.045
P_SEMI_TO_DISABLED = 0.10
DEATH_RATE = 0.05
NEW_ELDER_RATE = 0.07

# 附件2：月均服务需求
DEMAND_PER_PERSON = {
    "自理":   {"助餐": 14, "日间照料": 8,  "上门护理": 0,  "康复理疗": 2, "助浴": 0, "紧急救助": 0.15},
    "半失能": {"助餐": 20, "日间照料": 14, "上门护理": 6,  "康复理疗": 4, "助浴": 2, "紧急救助": 1},
    "失能":   {"助餐": 22, "日间照料": 18, "上门护理": 12, "康复理疗": 6, "助浴": 4, "紧急救助": 3},
}

# 附件2：单次营收
SERVICE_PRICE = {
    "助餐": 10,
    "日间照料": 20,
    "上门护理": 30,
    "康复理疗": 28,
    "助浴": 25,
    "紧急救助": 0,
}

# 附件2：单次直接支出
SERVICE_DIRECT_COST = {
    "助餐": 8,
    "日间照料": 16,
    "上门护理": 24,
    "康复理疗": 23,
    "助浴": 20,
    "紧急救助": 8,
}

# 月服务消费上限比例
CONSUME_LIMIT_RATIO = {
    "自理": 0.20,
    "半失能": 0.25,
    "失能": 0.30,
}

# 附件3：建设与运营成本
SCALE_INFO = {
    0: {"name": "不建", "build_cost": 0.0, "fixed_cost_day": 0,    "daily_capacity": 0},
    1: {"name": "小型", "build_cost": 18.0, "fixed_cost_day": 2000, "daily_capacity": 1000},
    2: {"name": "中型", "build_cost": 32.0, "fixed_cost_day": 3200, "daily_capacity": 2000},
    3: {"name": "大型", "build_cost": 45.0, "fixed_cost_day": 4400, "daily_capacity": 3000},
}

# 附件4：小区距离矩阵
DIST = {
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

# 参数
BUDGET = 120.0
SERVICE_RADIUS = 1000
ALPHA = 0.6
BETA = 0.4
DAYS_PER_MONTH = 30.0
MAX_ITER = 100
TOL = 1e-6
DAMPING = 0.5
MAX_UTIL_ALLOWED = 1.20


# 问题1：第5年末老人结构与消费约束需求

# 第5年末各类老人数量（来自Markov_output.txt）
ELDERLY_5YEAR = {
    "A": {"自理": 521, "半失能": 150, "失能": 114},
    "B": {"自理": 436, "半失能": 130, "失能": 106},
    "C": {"自理": 669, "半失能": 199, "失能": 150},
    "D": {"自理": 391, "半失能": 115, "失能": 94},
    "E": {"自理": 567, "半失能": 168, "失能": 129},
    "F": {"自理": 345, "半失能": 101, "失能": 74},
    "G": {"自理": 626, "半失能": 185, "失能": 142},
    "H": {"自理": 413, "半失能": 123, "失能": 91},
    "I": {"自理": 533, "半失能": 159, "失能": 120},
    "J": {"自理": 480, "半失能": 141, "失能": 105},
}

# 第5年末各类老人月度服务需求详情（考虑消费约束，来自Markov_output.txt）
DEMAND_DETAIL_5YEAR = {
    "A": {
        "自理": {"助餐": 7294, "日间照料": 4168, "上门护理": 0, "康复理疗": 1042, "助浴": 0, "紧急救助": 78},
        "半失能": {"助餐": 3000, "日间照料": 2100, "上门护理": 900, "康复理疗": 600, "助浴": 300, "紧急救助": 150},
        "失能": {"助餐": 2118, "日间照料": 1733, "上门护理": 1155, "康复理疗": 578, "助浴": 385, "紧急救助": 342},
    },
    "B": {
        "自理": {"助餐": 6104, "日间照料": 3488, "上门护理": 0, "康复理疗": 872, "助浴": 0, "紧急救助": 65},
        "半失能": {"助餐": 2451, "日间照料": 1716, "上门护理": 735, "康复理疗": 490, "助浴": 245, "紧急救助": 130},
        "失能": {"助餐": 1795, "日间照料": 1469, "上门护理": 979, "康复理疗": 490, "助浴": 326, "紧急救助": 318},
    },
    "C": {
        "自理": {"助餐": 9366, "日间照料": 5352, "上门护理": 0, "康复理疗": 1338, "助浴": 0, "紧急救助": 100},
        "半失能": {"助餐": 3980, "日间照料": 2786, "上门护理": 1194, "康复理疗": 796, "助浴": 398, "紧急救助": 199},
        "失能": {"助餐": 3114, "日间照料": 2548, "上门护理": 1699, "康复理疗": 849, "助浴": 566, "紧急救助": 450},
    },
    "D": {
        "自理": {"助餐": 5474, "日间照料": 3128, "上门护理": 0, "康复理疗": 782, "助浴": 0, "紧急救助": 59},
        "半失能": {"助餐": 2029, "日间照料": 1420, "上门护理": 609, "康复理疗": 406, "助浴": 203, "紧急救助": 115},
        "失能": {"助餐": 1489, "日间照料": 1219, "上门护理": 812, "康复理疗": 406, "助浴": 271, "紧急救助": 282},
    },
    "E": {
        "自理": {"助餐": 7938, "日间照料": 4536, "上门护理": 0, "康复理疗": 1134, "助浴": 0, "紧急救助": 85},
        "半失能": {"助餐": 3360, "日间照料": 2352, "上门护理": 1008, "康复理疗": 672, "助浴": 336, "紧急救助": 168},
        "失能": {"助餐": 2467, "日间照料": 2018, "上门护理": 1346, "康复理疗": 673, "助浴": 449, "紧急救助": 387},
    },
    "F": {
        "自理": {"助餐": 4830, "日间照料": 2760, "上门护理": 0, "康复理疗": 690, "助浴": 0, "紧急救助": 52},
        "半失能": {"助餐": 1659, "日间照料": 1161, "上门护理": 498, "康复理疗": 332, "助浴": 166, "紧急救助": 101},
        "失能": {"助餐": 1092, "日间照料": 893, "上门护理": 595, "康复理疗": 298, "助浴": 198, "紧急救助": 222},
    },
    "G": {
        "自理": {"助餐": 8764, "日间照料": 5008, "上门护理": 0, "康复理疗": 1252, "助浴": 0, "紧急救助": 94},
        "半失能": {"助餐": 3700, "日间照料": 2590, "上门护理": 1110, "康复理疗": 740, "助浴": 370, "紧急救助": 185},
        "失能": {"助餐": 2793, "日间照料": 2285, "上门护理": 1523, "康复理疗": 762, "助浴": 508, "紧急救助": 426},
    },
    "H": {
        "自理": {"助餐": 5782, "日间照料": 3304, "上门护理": 0, "康复理疗": 826, "助浴": 0, "紧急救助": 62},
        "半失能": {"助餐": 2245, "日间照料": 1571, "上门护理": 673, "康复理疗": 449, "助浴": 224, "紧急救助": 123},
        "失能": {"助餐": 1492, "日间照料": 1220, "上门护理": 814, "康复理疗": 407, "助浴": 271, "紧急救助": 273},
    },
    "I": {
        "自理": {"助餐": 7462, "日间照料": 4264, "上门护理": 0, "康复理疗": 1066, "助浴": 0, "紧急救助": 80},
        "半失能": {"助餐": 3180, "日间照料": 2226, "上门护理": 954, "康复理疗": 636, "助浴": 318, "紧急救助": 159},
        "失能": {"助餐": 2164, "日间照料": 1770, "上门护理": 1180, "康复理疗": 590, "助浴": 393, "紧急救助": 360},
    },
    "J": {
        "自理": {"助餐": 6720, "日间照料": 3840, "上门护理": 0, "康复理疗": 960, "助浴": 0, "紧急救助": 72},
        "半失能": {"助餐": 2745, "日间照料": 1921, "上门护理": 823, "康复理疗": 549, "助浴": 274, "紧急救助": 141},
        "失能": {"助餐": 1836, "日间照料": 1502, "上门护理": 1001, "康复理疗": 501, "助浴": 334, "紧急救助": 315},
    },
}

def get_monthly_demand():
    """计算每个小区的月度总服务需求（消费约束后）"""
    result = {}
    for c in COMMUNITIES:
        total = 0
        for et in ELDER_TYPES:
            total += sum(DEMAND_DETAIL_5YEAR[c][et].values())
        result[c] = total
    return result

def get_elderly_by_type():
    """获取各类老人数量"""
    return ELDERLY_5YEAR

def total_elderly_by_community():
    """获取每个小区的老人总数"""
    return {c: sum(ELDERLY_5YEAR[c].values()) for c in COMMUNITIES}


# 满意度与工具函数

def distance_satisfaction(d: float) -> float:
    if d <= 300:
        return 1.00
    elif d <= 500:
        return 0.90
    elif d <= 650:
        return 0.75
    elif d <= 1000:
        return 0.60
    else:
        return 0.0

def response_satisfaction(u: float) -> float:
    """
    方案2：
    当 u > 1 时，按日常经验取 0.30
    """
    if u <= 0.60:
        return 1.00
    elif u <= 0.75:
        return 0.93
    elif u <= 0.85:
        return 0.85
    elif u <= 0.95:
        return 0.72
    elif u <= 1.00:
        return 0.60
    else:
        return 0.30

def total_satisfaction(s1: float, s2: float) -> float:
    """
    问题二中 S3 = 1，因此：
    S = 0.2*S1 + 0.3*S2 + 0.5
    """
    if s1 <= 0:
        return 0.0
    return 0.2 * s1 + 0.3 * s2 + 0.5

def built_sites(plan: Dict[str, int]) -> List[str]:
    return [c for c in COMMUNITIES if plan[c] > 0]

def plan_cost(plan: Dict[str, int]) -> float:
    return sum(SCALE_INFO[plan[c]]["build_cost"] for c in COMMUNITIES)

def station_capacity(plan: Dict[str, int], site: str) -> int:
    return SCALE_INFO[plan[site]]["daily_capacity"]

def reachable_sites(plan: Dict[str, int], comm: str) -> List[str]:
    return [s for s in COMMUNITIES if plan[s] > 0 and DIST[comm][s] <= SERVICE_RADIUS]

def total_population(elderly_by_comm: Dict[str, int]) -> int:
    return sum(elderly_by_comm.values())


# 数据结构

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


# 固定点迭代

def fixed_point_allocation(plan: Dict[str, int],
                           monthly_demand: Dict[str, int],
                           elderly_by_comm: Dict[str, int]) -> AllocationResult:
    sites = built_sites(plan)

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
        # 1) 根据利用率计算响应满意度
        s2 = {s: response_satisfaction(util[s]) for s in sites}

        # 2) 各社区按满意度最大化选站
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

        # 3) 计算站点日负载
        station_load = {s: 0.0 for s in sites}
        for c in COMMUNITIES:
            s = assignments[c]
            if s is None:
                continue
            station_load[s] += monthly_demand[c] * community_sat[c] / DAYS_PER_MONTH

        # 4) 更新利用率
        new_util = {}
        for s in sites:
            cap = station_capacity(plan, s)
            new_util[s] = station_load[s] / cap if cap > 0 else 0.0

        # 5) 阻尼固定点更新
        updated_util = {}
        for s in sites:
            updated_util[s] = DAMPING * new_util[s] + (1 - DAMPING) * util[s]

        max_diff = max(abs(updated_util[s] - util[s]) for s in sites)
        same_assign = (assignments == prev_assignments)

        util = updated_util
        prev_assignments = assignments

        if same_assign and max_diff < TOL:
            break

    # 最终重算
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
        if s is None:
            continue
        final_station_load[s] += monthly_demand[c] * final_community_sat[c] / DAYS_PER_MONTH

    final_station_util = {}
    for s in sites:
        cap = station_capacity(plan, s)
        final_station_util[s] = final_station_load[s] / cap if cap > 0 else 0.0

    # 方案2：允许轻度超载，但超过1.20判不可行
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
    coverage_rate = covered_pop / total_population(elderly_by_comm)

    avg_satisfaction = sum(
        elderly_by_comm[c] * final_community_sat[c] for c in COMMUNITIES
    ) / total_population(elderly_by_comm)

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


# 分支定界

def optimistic_upper_bound(node: Node) -> float:
    """
    保守上界：目标值不超过1
    """
    return 1.0

def branch_and_bound(monthly_demand: Dict[str, int],
                     elderly_by_comm: Dict[str, int]) -> Tuple[BestSolution, List[str]]:
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

    # 按需求高低排序
    order = sorted(COMMUNITIES, key=lambda c: monthly_demand[c], reverse=True)

    def dfs(node: Node):
        nonlocal best

        # 预算剪枝
        if node.cost > BUDGET + 1e-12:
            return

        # 上界剪枝
        ub = optimistic_upper_bound(node)
        if ub <= best.result.objective_value + 1e-12:
            return

        # 完整方案评估
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

        # 分支顺序：大型 -> 中型 -> 小型 -> 不建
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


# 年利润计算

def annualized_build_cost(scale: int) -> float:
    """
    建设成本按20年平均折旧，单位：元/年
    """
    return SCALE_INFO[scale]["build_cost"] * 10000 / 20.0

def estimate_station_profit(plan: Dict[str, int],
                            alloc_result: AllocationResult,
                            demand_detail_after_constraint: Dict[str, Dict[str, Dict[str, int]]]):
    """
    计算每个站点预计年度利润
    假设：
    - 社区若分配到某站，则其6类服务均由该站承担
    - 有效服务人次 = 理论需求 * 社区满意度
    """
    station_profit = {}

    for s in built_sites(plan):
        annual_revenue = 0.0
        annual_direct_cost = 0.0

        for c in COMMUNITIES:
            if alloc_result.assignments[c] != s:
                continue

            sat = alloc_result.community_sat[c]

            # 该社区的所有服务项目由站点s提供
            for et in ELDER_TYPES:
                for srv in SERVICE_TYPES:
                    monthly_effective_qty = demand_detail_after_constraint[c][et][srv] * sat
                    annual_revenue += monthly_effective_qty * SERVICE_PRICE[srv] * 12
                    annual_direct_cost += monthly_effective_qty * SERVICE_DIRECT_COST[srv] * 12

        annual_fixed_cost = SCALE_INFO[plan[s]]["fixed_cost_day"] * 365.0
        annual_build_cost = annualized_build_cost(plan[s])

        annual_profit = annual_revenue - annual_direct_cost - annual_fixed_cost - annual_build_cost

        station_profit[s] = {
            "annual_revenue": annual_revenue,
            "annual_direct_cost": annual_direct_cost,
            "annual_fixed_cost": annual_fixed_cost,
            "annual_build_cost": annual_build_cost,
            "annual_profit": annual_profit
        }

    return station_profit


# 输出

def print_solution(best: BestSolution,
                   order: List[str],
                   elderly_5y: Dict[str, Dict[str, int]],
                   monthly_demand: Dict[str, int],
                   station_profit: Dict[str, Dict[str, float]]):

    elderly_by_comm = total_elderly_by_community()

    print("=" * 100)
    print("问题二最优解（方案2：S2超载取0.30，最大利用率1.20）")
    print("=" * 100)

    print("一、第5年末各社区老人数量")
    for c in COMMUNITIES:
        print(f"  社区{c}: 总数={elderly_by_comm[c]}, 结构={elderly_5y[c]}")
    print()

    print("二、消费约束后的月需求总量")
    for c in COMMUNITIES:
        print(f"  社区{c}: {monthly_demand[c]} 次/月")
    print()

    print("三、候选点决策顺序（按需求降序）")
    print(" ", order)
    print()

    print("四、最优建站方案")
    print(f"  预算上限: {BUDGET:.2f} 万元")
    print(f"  实际建设成本: {best.cost:.2f} 万元")
    print(f"  服务站数量: {len(built_sites(best.plan))}")
    for s in built_sites(best.plan):
        print(f"  站点{s}: {SCALE_INFO[best.plan[s]]['name']}")
    print()

    print("五、覆盖率与满意度")
    print(f"  覆盖率: {best.result.coverage_rate:.4%}")
    print(f"  平均满意度: {best.result.avg_satisfaction:.6f}")
    print(f"  综合目标值: {best.result.objective_value:.6f}")
    print()

    print("六、社区分配结果")
    for c in COMMUNITIES:
        print(f"  社区{c} -> {best.result.assignments[c]}, 满意度={best.result.community_sat[c]:.4f}")
    print()

    print("七、站点负载、利用率与响应满意度")
    for s in built_sites(best.plan):
        print(
            f"  站点{s}: "
            f"日负载={best.result.station_load[s]:.2f}, "
            f"利用率={best.result.station_util[s]:.4f}, "
            f"S2={best.result.station_s2[s]:.2f}"
        )
    print()

    print("八、每个服务站预计年度利润")
    for s in built_sites(best.plan):
        info = station_profit[s]
        print(
            f"  站点{s}: "
            f"年收入={info['annual_revenue']:.2f} 元, "
            f"年直接支出={info['annual_direct_cost']:.2f} 元, "
            f"年固定成本={info['annual_fixed_cost']:.2f} 元, "
            f"年折旧成本={info['annual_build_cost']:.2f} 元, "
            f"预计年利润={info['annual_profit']:.2f} 元"
        )
    print("=" * 100)


# 主程序

def solve_problem2():
    # 直接使用Markov_output.txt中的数据
    elderly_5y = get_elderly_by_type()
    monthly_demand = get_monthly_demand()
    elderly_by_comm = total_elderly_by_community()

    # 求解问题2
    best, order = branch_and_bound(monthly_demand, elderly_by_comm)

    # 利润计算（使用硬编码的需求详情）
    station_profit = estimate_station_profit(best.plan, best.result, DEMAND_DETAIL_5YEAR)

    # 输出
    print_solution(best, order, elderly_5y, monthly_demand, station_profit)
    import sys
    sys.stdout.flush()


if __name__ == "__main__":
    solve_problem2()
