import math

class Config:
  

# ================= 集群环境配置 =================
    NUM_NODES = 12  # 修改为 12 个节点
    GPUS_PER_NODE = 4
    TOTAL_GPUS = 48  # 修改为 48 块卡
    NODE_CONFIGS = [
        # 高性能节点：3个 (共12张卡)
        {"count": 3, "type": "high", "cap": 1.8, "vram": 80, "interconnect": "nvlink"},
        # 通用节点：6个 (共24张卡)
        {"count": 6, "type": "mid",  "cap": 1.0, "vram": 40, "interconnect": "pcie"},
        # 入门节点：3个 (共12张卡)
        {"count": 3, "type": "low",  "cap": 0.55, "vram": 16, "interconnect": "pcie"}
    ]


    # ================= 负载生成配置 =================
    NUM_TASKS = 600
    ARRIVAL_RATE_LAMBDA = 4.0  # 泊松到达率 (任务/小时)
    DURATION_MEDIAN_H = 3.2    # 中位数运行时间 (小时)
    DURATION_P90_H = 18.0      # 90分位数运行时间 (小时)
    # 对数正态分布参数计算 (基于中位数和P90)
    MU = math.log(DURATION_MEDIAN_H)
    SIGMA = (math.log(DURATION_P90_H) - MU) / 1.2816 
    
    TASK_PROPS = {
        "has_deadline": 0.20,
        "high_priority": 0.15,
        "has_checkpoint": 0.70,
        "is_elastic": 0.35
    }

    # ================= 预测与误差配置 =================
    PRED_ERROR_MU = 0.0
    PRED_ERROR_SIGMA = 0.3    # 预测误差对数正态分布的标准差
    SAFETY_BUFFER_PERCENTILE = 0.90  # 回填安全缓冲取90分位误差

    # ================= 调度权重配置 =================
    WEIGHTS = {
        "base": 1.0,
        "value": 10.0,
        "urgency": 20.0,
        "aging": 5.0,
        "fitness": 8.0,
        "fairness": 12.0,
        "penalty": 5.0
    }
    FAIRNESS_LAMBDA = 0.15  # tanh补偿系数
    SLIDING_WINDOW_H = 24   # 公平性滑动窗口 (小时)

    # ================= 目标份额与课题组 =================
    NUM_GROUPS = 10

    # ================= 仿真控制 =================
    TIME_STEP_MIN = 1  # 仿真步长(分钟)
    MAX_SIM_TIME_MIN = 60000  # 防死循环