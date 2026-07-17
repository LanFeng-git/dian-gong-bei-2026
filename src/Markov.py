import pandas as pd
import numpy as np
import re
import os

# 获取数据目录（项目根目录下的 data/）
script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

# 数值提取函数
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

# 读取附件1
df1 = pd.read_excel(os.path.join(script_dir, '附件1：小区基础数据.xlsx'), sheet_name='人口与老人结构', header=1)
communities = df1.iloc[:, 0].values
N_self_init = df1.iloc[:, 3].values.astype(float)
N_semi_init = df1.iloc[:, 4].values.astype(float)
N_dis_init  = df1.iloc[:, 5].values.astype(float)
income      = df1.iloc[:, 6].values.astype(float)

# 读取附件2
df2_req = pd.read_excel(os.path.join(script_dir, '附件2：服务需求数据.xlsx'), sheet_name='每位老人月均服务需求次数', header=1)
q_self = df2_req.iloc[:, 1].values.astype(float)
q_semi = df2_req.iloc[:, 2].values.astype(float)
q_dis  = df2_req.iloc[:, 3].values.astype(float)
services = df2_req.iloc[:, 0].values

df2_price = pd.read_excel('附件2：服务需求数据.xlsx', sheet_name='服务营收及支出', header=1)
revenue_raw = df2_price.iloc[:, 1].values
base_price = np.array([extract_number(v) for v in revenue_raw])

df2_limit = pd.read_excel('附件2：服务需求数据.xlsx', sheet_name='月服务消费上限', header=0)
beta_raw = df2_limit.iloc[:, 1].values
beta = np.array([extract_number(v) for v in beta_raw])

# 马尔可夫人口预测
def markov_population_yearly(s, se, d, years=5, mu=0.05, lam=0.07, p12=0.045, p23=0.10):
    """返回列表，每个元素为 (self, semi, dis) 表示当年年末人数"""
    history = [(int(s), int(se), int(d))]
    for _ in range(years):
        total = s + se + d
        s_new = s * (1 - mu) * (1 - p12) + lam * total
        se_new = se * (1 - mu) * (1 - p23) + s * (1 - mu) * p12
        d_new = d * (1 - mu) + se * (1 - mu) * p23
        s, se, d = round(s_new), round(se_new), round(d_new)
        history.append((int(s), int(se), int(d)))
    return history

# 存储各小区每年数据
yearly_pop = []
for i in range(len(communities)):
    hist = markov_population_yearly(N_self_init[i], N_semi_init[i], N_dis_init[i])
    yearly_pop.append(hist)

# 提取第5年末数据
N_self_5 = [hist[5][0] for hist in yearly_pop]
N_semi_5 = [hist[5][1] for hist in yearly_pop]
N_dis_5  = [hist[5][2] for hist in yearly_pop]

# 输出第1年至第5年末各小区人口
print("=" * 80)
print("【1】第1年至第5年末各小区各类老人数量")
print("小区\t年份\t自理\t半失能\t失能\t合计")
for i, comm in enumerate(communities):
    for t in range(1, 6):
        self_t, semi_t, dis_t = yearly_pop[i][t]
        total_t = self_t + semi_t + dis_t
        print(f"{comm}\t第{t}年末\t{self_t}\t{semi_t}\t{dis_t}\t{total_t}")
    print()

# 理论月需求
print("\n" + "=" * 80)
print("【2】第5年末每个小区各项服务的理论月需求（分自理、半失能、失能，未考虑消费约束）")
print("小区\t老人类型\t服务项目\t理论月需求(次/月)")
for i, comm in enumerate(communities):
    self_cnt = N_self_5[i]
    semi_cnt = N_semi_5[i]
    dis_cnt  = N_dis_5[i]
    for k, (cnt, q_arr) in enumerate([(self_cnt, q_self), (semi_cnt, q_semi), (dis_cnt, q_dis)]):
        type_name = ["自理", "半失能", "失能"][k]
        for m, svc in enumerate(services):
            demand = cnt * q_arr[m]
            demand_int = int(round(demand))
            print(f"{comm}\t{type_name}\t{svc}\t{demand_int}")

# 消费约束削减系数
base_price_charge = base_price[:5]
q_self_charge = q_self[:5]
q_semi_charge = q_semi[:5]
q_dis_charge  = q_dis[:5]
Q = np.vstack([q_self_charge, q_semi_charge, q_dis_charge])
Pop5_mat = np.column_stack([N_self_5, N_semi_5, N_dis_5])

# 每类老人理论月消费总额
E = np.sum(Q * base_price_charge, axis=1)

# 削减系数
alpha = np.ones((len(communities), 3))
for i, inc in enumerate(income):
    for k in range(3):
        limit = inc * beta[k]
        if E[k] > limit:
            alpha[i, k] = limit / E[k]

# 计算约束后需求
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

# 输出约束后需求
print("\n" + "=" * 80)
print("【3】第5年末每个小区各类老人月均服务需求次数（考虑消费约束，等比例削减取整）")
print("小区\t老人类型\t服务项目\t约束后月需求(次/月)")
for i, comm in enumerate(communities):
    for k, type_name in enumerate(["自理", "半失能", "失能"]):
        demand_k = constrained_demand[i][k]
        for m, svc in enumerate(services):
            print(f"{comm}\t{type_name}\t{svc}\t{demand_k[m]}")

print("\n代码执行完成！")
