import data_generator
import simulator
import json

def run_strategy(strategy_name, ablations={}):
    print(f"运行策略: {strategy_name} (消融: {ablations})...")
    sim = simulator.Simulator(strategy_name, ablations)
    finished_tasks, end_time = sim.run()
    metrics = sim.evaluate(finished_tasks, end_time)
    metrics["Strategy"] = strategy_name
    return metrics

if __name__ == "__main__":
    print("===== 步骤 1: 生成固定集群环境与任务轨迹 =====")
    data_generator.generate_cluster_env()
    data_generator.generate_workload(seed=42)
    
    results = []
    
    print("\n===== 步骤 2: 运行基础对比实验 =====")
    results.append(run_strategy("FCFS"))
    results.append(run_strategy("Static"))
    results.append(run_strategy("PCBP"))
    
    print("\n===== 步骤 3: 运行消融实验 =====")
    results.append(run_strategy("PCBP (No Aging)", {"aging": True}))
    results.append(run_strategy("PCBP (No Fitness)", {"fitness": True}))
    results.append(run_strategy("PCBP (No Backfill)", {"backfill": True}))
    
    print("\n===== 仿真实验结果汇总 =====")
    headers = ["Strategy", "Avg Wait (h)", "95p Wait (h)", "Util Apparent (%)", "Util Effective (%)", "Viol Rate (%)", "High Val Comp (%)", "Jain Index"]
    widths = [22, 14, 14, 20, 20, 16, 18, 12]
    
    header_row = "".join(f"{h:<{w}}" for h, w in zip(headers, widths))
    print(header_row)
    print("-" * len(header_row))
    
    for res in results:
        row_data = [res[h] for h in headers]
        print("".join(f"{str(c):<{w}}" for c, w in zip(row_data, widths)))