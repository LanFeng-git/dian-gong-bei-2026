# -*- coding: utf-8 -*-
"""
问题三：服务定价与政府补贴优化
依赖：pandas, openpyxl

运行：把本脚本与以下文件放在同一目录后执行：
python problem3.py

需要文件：
- Markov_output.txt
- BnB_output.txt
- 附件1：小区基础数据.xlsx
- 附件2：服务需求数据.xlsx
- 附件3：服务站建设与运营成本.xlsx

输出：
- 问题三_定价补贴优化结果.xlsx
"""

from pathlib import Path
import re
import math
import sys
import io
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent  # 项目根目录（src/ 的上层）
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

MARKOV_TXT = RESULTS_DIR / "Markov_output.txt"
BNB_TXT = RESULTS_DIR / "BnB_output.txt"
ATTACH1 = DATA_DIR / "附件1：小区基础数据.xlsx"
ATTACH2 = DATA_DIR / "附件2：服务需求数据.xlsx"
ATTACH3 = DATA_DIR / "附件3：服务站建设与运营成本.xlsx"

OUT_XLSX = RESULTS_DIR / "问题三_定价补贴优化结果.xlsx"
OUT_TXT = RESULTS_DIR / "Gsm_output.txt"

SERVICE_ORDER = ["助餐", "日间照料", "上门护理", "康复理疗", "助浴", "紧急救助"]
NON_EMERGENCY = [s for s in SERVICE_ORDER if s != "紧急救助"]
COMMUNITIES = list("ABCDEFGHIJ")

PROFIT_RATE_UPPER = 0.08
SUBSIDY_PER_PERSON_TIME = 2.0
DAYS = 365


def read_text_auto(path: Path) -> str:
    """自动识别常见中文/UTF编码读取txt。"""
    for enc in ("utf-8-sig", "utf-16", "utf-16-le", "gb18030", "utf-8"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeError:
            continue
    raise UnicodeError(f"无法识别文本编码：{path}")


def to_float(x):
    """将'0（公益免费）'、'≤ 20%'等文本尽量转成数字。"""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    m = re.search(r"-?\d+(?:\.\d+)?", str(x))
    return float(m.group()) if m else np.nan


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def parse_markov_output(path: Path):
    """从 Markov_output.txt 解析第5年末老人数量和消费约束后的月需求量。"""
    text = read_text_auto(path)

    sec1 = text[text.index("【1】"):text.index("【2】")]
    pop_rows = []

    for line in sec1.splitlines():
        parts = line.strip().split()
        if len(parts) >= 6 and parts[0] in COMMUNITIES and parts[1] == "第5年末":
            pop_rows.append({
                "小区": parts[0],
                "自理": int(float(parts[2])),
                "半失能": int(float(parts[3])),
                "失能": int(float(parts[4])),
                "合计": int(float(parts[5])),
            })

    pop_df = pd.DataFrame(pop_rows)

    sec3 = text[text.index("【3】"):]
    if "代码执行完成" in sec3:
        sec3 = sec3[:sec3.index("代码执行完成")]

    demand_rows = []
    pattern = re.compile(r"^([A-J])\s+(自理|半失能|半自理|失能)\s+(\S+)\s+(\d+(?:\.\d+)?)")

    for line in sec3.splitlines():
        m = pattern.match(line.strip())
        if not m:
            continue

        comm, elder_type, service, value = m.groups()
        elder_type = "半失能" if elder_type == "半自理" else elder_type

        if service not in SERVICE_ORDER:
            continue

        demand_rows.append({
            "小区": comm,
            "老人类型": elder_type,
            "服务项目": service,
            "约束后月需求": int(float(value)),
        })

    demand_df = pd.DataFrame(demand_rows)

    if pop_df.empty:
        raise ValueError("未能解析第5年末老人数量，请检查 Markov_output.txt 格式。")
    if demand_df.empty:
        raise ValueError("未能解析消费约束后的服务需求，请检查 Markov_output.txt 格式。")

    return pop_df, demand_df


def parse_bnb_output(path: Path):
    """从 BnB_output.txt 解析问题二最优方案、社区分配和满意度。"""
    text = read_text_auto(path)

    station_sizes = dict(re.findall(r"站点([A-J]):\s*(小型|中型|大型)", text))

    assign_rows = []
    for comm, site, sat in re.findall(r"社区([A-J])\s*->\s*([A-J]),\s*满意度=([0-9.]+)", text):
        assign_rows.append({
            "小区": comm,
            "服务站": site,
            "综合满意度_问题二": float(sat),
        })

    assign_df = pd.DataFrame(assign_rows)

    load_rows = []
    for site, daily_load, util, s2 in re.findall(
        r"站点([A-J]):\s*日负载=([0-9.]+),\s*利用率=([0-9.]+),\s*S2=([0-9.]+)",
        text,
    ):
        load_rows.append({
            "服务站": site,
            "日负载": float(daily_load),
            "利用率": float(util),
            "响应满意度S2": float(s2),
        })

    load_df = pd.DataFrame(load_rows)

    coverage = re.search(r"覆盖率:\s*([0-9.]+)%", text)
    avg_sat = re.search(r"平均满意度:\s*([0-9.]+)", text)

    summary = {
        "覆盖率": float(coverage.group(1)) / 100 if coverage else np.nan,
        "平均满意度": float(avg_sat.group(1)) if avg_sat else np.nan,
    }

    if not station_sizes:
        raise ValueError("未能解析服务站位置与规模，请检查 BnB_output.txt。")
    if assign_df.empty:
        raise ValueError("未能解析社区分配，请检查 BnB_output.txt。")

    return station_sizes, assign_df, load_df, summary


def load_service_params(path: Path):
    """读取附件2：服务基准价与直接支出。"""
    df = pd.read_excel(path, sheet_name="服务营收及支出", header=1)
    df = clean_columns(df).dropna(subset=["服务项目"])
    df = df[df["服务项目"].isin(SERVICE_ORDER)].copy()

    revenue_col = [c for c in df.columns if "营收" in c][0]
    cost_col = [c for c in df.columns if "直接支出" in c][0]

    df["基准价格"] = df[revenue_col].apply(to_float).fillna(0.0)
    df["单次直接支出"] = df[cost_col].apply(to_float).fillna(0.0)

    base_price = dict(zip(df["服务项目"], df["基准价格"]))
    direct_cost = dict(zip(df["服务项目"], df["单次直接支出"]))

    return base_price, direct_cost, df[["服务项目", "基准价格", "单次直接支出"]]


def load_station_params(path: Path):
    """读取附件3：建设成本、日固定管理成本、日最大服务人次。"""
    df = pd.read_excel(path, sheet_name="服务站建设与运营成本", header=1)
    df = clean_columns(df).dropna(subset=["站点规模"])
    df = df[df["站点规模"].isin(["小型", "中型", "大型"])].copy()

    build_col = [c for c in df.columns if "一次性建设成本" in c][0]
    fixed_col = [c for c in df.columns if "日均固定管理成本" in c][0]
    cap_col = [c for c in df.columns if "日最大服务人次" in c][0]

    df["一次性建设成本_元"] = df[build_col].apply(to_float) * 10000
    df["年固定管理成本"] = df[fixed_col].apply(to_float) * DAYS
    df["年折旧成本"] = df["一次性建设成本_元"] / 20.0
    df["日最大服务人次"] = df[cap_col].apply(to_float)

    df["日补贴上限"] = df["站点规模"].map({
        "小型": 1000,
        "中型": 1800,
        "大型": 2600,
    })
    df["年补贴上限"] = df["日补贴上限"] * DAYS

    return df.set_index("站点规模")


def price_satisfaction(price, base_price):
    """附件5价格满意度规则。"""
    if base_price <= 0:
        return 1.0 if price <= 0 else 0.6

    ratio = price / base_price

    if ratio <= 1.0:
        return 1.00
    if ratio <= 1.10:
        return 0.90
    if ratio <= 1.20:
        return 0.75
    return 0.60


def ceil_to_cent(x):
    """向上保留2位小数，避免定价舍入后导致亏损。"""
    return math.ceil((x - 1e-12) * 100) / 100


def main():
    # 捕获控制台输出
    output_buffer = []
    
    def capture_print(*args, sep=' ', end='\n', file=None, flush=False):
        output_buffer.append(sep.join(str(arg) for arg in args) + end)
    
    old_print = print
    __builtins__.print = capture_print
    
    pop_df, demand_df = parse_markov_output(MARKOV_TXT)
    station_sizes, assign_df, load_df, bnb_summary = parse_bnb_output(BNB_TXT)
    base_price, direct_cost, service_param_df = load_service_params(ATTACH2)
    station_param_df = load_station_params(ATTACH3)

    income_df = pd.read_excel(ATTACH1, sheet_name="人口与老人结构", header=1)
    income_df = clean_columns(income_df).dropna(subset=["小区编号"])
    income_df = income_df.rename(columns={"小区编号": "小区"})
    income_col = [c for c in income_df.columns if "收入" in c][0]
    income_df = income_df[["小区", income_col]].rename(columns={income_col: "人均月收入"})

    df = demand_df.merge(assign_df, on="小区", how="left")

    df["实际有效月需求"] = df["约束后月需求"] * df["综合满意度_问题二"]
    df["实际有效年需求"] = df["实际有效月需求"] * 12

    df["基准价格"] = df["服务项目"].map(base_price)
    df["单次直接支出"] = df["服务项目"].map(direct_cost)

    df["基准年收入"] = df["实际有效年需求"] * df["基准价格"]
    df["年直接支出"] = df["实际有效年需求"] * df["单次直接支出"]

    site_service = (
        df.pivot_table(
            index="服务站",
            columns="服务项目",
            values="实际有效年需求",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(columns=SERVICE_ORDER, fill_value=0)
        .reset_index()
    )

    pricing_rows = []
    profit_rows = []

    for site, size in station_sizes.items():
        sub = df[df["服务站"] == site].copy()
        sp = station_param_df.loc[size]

        base_revenue = sub["基准年收入"].sum()
        direct_total = sub["年直接支出"].sum()

        fixed_cost = float(sp["年固定管理成本"])
        depreciation = float(sp["年折旧成本"])
        total_cost = direct_total + fixed_cost + depreciation

        non_emergency_qty = sub.loc[
            sub["服务项目"] != "紧急救助",
            "实际有效年需求",
        ].sum()

        subsidy_raw = SUBSIDY_PER_PERSON_TIME * non_emergency_qty
        subsidy = min(subsidy_raw, float(sp["年补贴上限"]))

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
            "年补贴上限": float(sp["年补贴上限"]),
            "实际政府补贴": subsidy,
            "预计年利润": profit,
            "利润率": profit_rate,
            "是否满足利润率≤8%": profit_rate <= PROFIT_RATE_UPPER,
            "是否保本": profit >= -1e-6,
        })

    pricing_long = pd.DataFrame(pricing_rows)
    profit_df = pd.DataFrame(profit_rows).sort_values("服务站")

    pricing_wide = (
        pricing_long.pivot_table(
            index=["服务站", "规模", "折扣系数rho"],
            columns="服务项目",
            values="最优定价",
            aggfunc="first",
        )
        .reindex(columns=SERVICE_ORDER)
        .reset_index()
    )

    price_s3_map = pricing_long.set_index(["服务站", "服务项目"])["价格满意度S3"].to_dict()

    tmp = df.copy()
    tmp["价格满意度S3"] = tmp.apply(
        lambda r: price_s3_map[(r["服务站"], r["服务项目"])],
        axis=1,
    )

    comm_price_sat = (
        tmp.assign(weight=tmp["实际有效年需求"])
        .groupby("小区")
        .apply(
            lambda g: np.average(g["价格满意度S3"], weights=g["weight"])
            if g["weight"].sum() > 0 else np.nan,
            include_groups=False,
        )
        .reset_index(name="价格满意度S3")
    )

    comm_sat_df = (
        assign_df.merge(comm_price_sat, on="小区", how="left")
        .rename(columns={"综合满意度_问题二": "综合满意度"})
        .sort_values("小区")
    )

    price_map = pricing_long.set_index(["服务站", "服务项目"])["最优定价"].to_dict()

    df["优化后年支付"] = df.apply(
        lambda r: r["实际有效年需求"] * price_map[(r["服务站"], r["服务项目"])],
        axis=1,
    )
    df["基准年支付"] = df["基准年收入"]

    pop_long = pop_df.melt(
        id_vars="小区",
        value_vars=["自理", "半失能", "失能"],
        var_name="老人类型",
        value_name="人数",
    )

    pop_income = pop_long.merge(income_df, on="小区", how="left")

    access_rows = []

    for elder_type in ["自理", "半失能", "失能"]:
        n = pop_long.loc[pop_long["老人类型"] == elder_type, "人数"].sum()

        base_month = (
            df.loc[df["老人类型"] == elder_type, "基准年支付"].sum()
            / 12
            / n
        )

        opt_month = (
            df.loc[df["老人类型"] == elder_type, "优化后年支付"].sum()
            / 12
            / n
        )

        tmp_income = pop_income[pop_income["老人类型"] == elder_type]
        income_weighted = (
            tmp_income["人数"] * tmp_income["人均月收入"]
        ).sum() / n

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

    print("\n========== 问题三：最优定价（元/次） ==========")
    print(pricing_wide.round(4).to_string(index=False))

    print("\n========== 站点利润与补贴 ==========")
    cols = [
        "服务站",
        "规模",
        "优化后年服务收入",
        "年度总成本",
        "实际政府补贴",
        "预计年利润",
        "利润率",
        "是否满足利润率≤8%",
    ]
    print(profit_df[cols].round(4).to_string(index=False))

    print("\n========== 社区满意度 ==========")
    print(comm_sat_df.round(4).to_string(index=False))

    print("\n========== 类型可及性 ==========")
    print(access_df.round(4).to_string(index=False))

    print("\n问题二覆盖率：", bnb_summary["覆盖率"])
    print("问题二平均满意度：", bnb_summary["平均满意度"])

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        service_param_df.to_excel(writer, sheet_name="输入_服务价格成本", index=False)
        pd.DataFrame([
            {"服务站": k, "规模": v}
            for k, v in station_sizes.items()
        ]).to_excel(writer, sheet_name="输入_固定站点方案", index=False)

        assign_df.to_excel(writer, sheet_name="输入_社区分配", index=False)
        site_service.to_excel(writer, sheet_name="站点服务量", index=False)
        pricing_long.to_excel(writer, sheet_name="定价_long", index=False)
        pricing_wide.to_excel(writer, sheet_name="定价_wide", index=False)
        profit_df.to_excel(writer, sheet_name="利润补贴", index=False)
        comm_sat_df.to_excel(writer, sheet_name="社区满意度", index=False)
        access_df.to_excel(writer, sheet_name="类型可及性", index=False)

    print(f"\n结果已保存：{OUT_XLSX}")
    
    # 恢复print并获取输出内容
    __builtins__.print = old_print
    output_content = ''.join(output_buffer)
    
    # 保存输出内容到文本文件
    OUT_TXT.write_text(output_content, encoding="utf-8-sig")
    print(f"输出内容已保存：{OUT_TXT}")


if __name__ == "__main__":
    main()
