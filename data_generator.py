import json
import math
import random
from config import Config

def generate_cluster_env():
    cluster = []
    gpu_id = 0
    for cfg in Config.NODE_CONFIGS:
        for _ in range(cfg["count"]):
            node = {
                "node_id": len(cluster),
                "gpus": [],
                "type": cfg["type"]
            }
            for _ in range(Config.GPUS_PER_NODE):
                node["gpus"].append({
                    "gpu_id": gpu_id,
                    "cap": cfg["cap"],
                    "vram": cfg["vram"],
                    "interconnect": cfg["interconnect"]
                })
                gpu_id += 1
            cluster.append(node)
    with open("cluster_env.json", "w") as f:
        json.dump(cluster, f, indent=4)
    print("集群环境已生成: cluster_env.json")

def generate_workload(seed=42):
    random.seed(seed)
    tasks = []
    current_arrival = 0.0
    
    for i in range(Config.NUM_TASKS):
        # 到达时间 (分钟)
        current_arrival += random.expovariate(Config.ARRIVAL_RATE_LAMBDA) * 60
        arrival_time = current_arrival
        
        # GPU需求
        rand_g = random.random()
        if rand_g < 0.4: num_gpus = 1
        elif rand_g < 0.7: num_gpus = 2
        elif rand_g < 0.9: num_gpus = 4
        else: num_gpus = 8
        
        # 运行时间 (分钟) 与 显存需求 (GB)
        duration_h = random.lognormvariate(Config.MU, Config.SIGMA)
        duration_h = max(0.1, min(duration_h, 100.0))
        duration_min = duration_h * 60
        
        # 显存需求根据GPU数量和类型粗略估算
        mem_per_gpu = random.choice([4, 8, 12, 16, 24, 40, 60])
        
        # 预测时间与真实时间的误差
        error_factor = math.exp(random.gauss(Config.PRED_ERROR_MU, Config.PRED_ERROR_SIGMA))
        predicted_duration = duration_min * error_factor
        
        # 任务属性
        has_deadline = random.random() < Config.TASK_PROPS["has_deadline"]
        if has_deadline:
            deadline = arrival_time + duration_min + random.uniform(360, 1440)
        else:
            deadline = float('inf')
            
        task = {
            "task_id": i,
            "group_id": random.randint(0, Config.NUM_GROUPS - 1),
            "arrival_time": arrival_time,
            "num_gpus": num_gpus,
            "duration": duration_min,             # 真实运行时间
            "predicted_duration": predicted_duration, # 预测运行时间
            "mem_per_gpu": mem_per_gpu,
            "deadline": deadline,
            "priority": 1 if random.random() < Config.TASK_PROPS["high_priority"] else 0,
            "has_ckpt": random.random() < Config.TASK_PROPS["has_checkpoint"],
            "is_elastic": random.random() < Config.TASK_PROPS["is_elastic"]
        }
        tasks.append(task)
        
    with open("workload_tasks.json", "w") as f:
        json.dump(tasks, f, indent=4)
    print(f"任务负载已生成: workload_tasks.json (Seed: {seed}, 任务数: {len(tasks)})")

if __name__ == "__main__":
    generate_cluster_env()
    generate_workload(seed=42)