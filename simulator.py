import json
import math
import random
from config import Config

class GPU:
    def __init__(self, gpu_id, cap, vram, node_id):
        self.id = gpu_id
        self.cap = cap
        self.vram = vram
        self.node_id = node_id
        self.busy_until = 0
        self.alloc_task = None

class Task:
    def __init__(self, data):
        self.__dict__.update(data)
        self.start_time = None
        self.finish_time = None
        self.alloc_gpus = []
        self.dynamic_p = 0.0
        
class Scheduler:
    def __init__(self, gpus, nodes, strategy="FCFS", ablations={}):
        self.gpus = gpus
        self.nodes = nodes
        self.strategy = strategy
        self.ablations = ablations
        
        self.group_gpu_hours = {i: 0.0 for i in range(Config.NUM_GROUPS)}
        self.group_target_share = {i: 1.0/Config.NUM_GROUPS for i in range(Config.NUM_GROUPS)}
        
    def calculate_fitness(self, task, candidate_gpus):
        if len(candidate_gpus) < task.num_gpus: return -1, 0
        selected = candidate_gpus[:task.num_gpus]
        min_vram = min(g.vram for g in selected)
        if min_vram < task.mem_per_gpu: return -1, 0
        
        fitness = 10.0
        avg_vram = sum(g.vram for g in selected) / len(selected)
        avg_cap = sum(g.cap for g in selected) / len(selected)
        
        if task.mem_per_gpu > avg_vram * 0.7: fitness -= 8.0
        if task.num_gpus > 2 and avg_cap < 1.0: fitness -= 5.0
        if avg_vram > task.mem_per_gpu * 3: fitness -= 2.0
            
        n = task.num_gpus
        s_n = 1.0
        if n > 1:
            s_n = 1.0 / (1 + 0.15 * (n - 1))
            node_set = set(g.node_id for g in selected)
            if len(node_set) > 1: s_n *= 0.8 ** (len(node_set) - 1)
            fitness += s_n * 5.0
        return fitness, s_n

    def get_priority(self, task, current_time):
        if self.strategy == "FCFS": return task.arrival_time 
        if self.strategy == "Static": return (-task.priority, task.arrival_time)
            
        w = Config.WEIGHTS
        B = 1.0
        V = task.priority * w["value"]
        if task.deadline != float('inf'):
            slack_h = (task.deadline - current_time - task.duration) / 60
            U = 1 / (1 + math.exp(slack_h)) * w["urgency"]
        else: U = 0
            
        if "aging" in self.ablations and self.ablations["aging"]: A = 0
        else:
            wait_h = (current_time - task.arrival_time) / 60
            A = math.log(wait_h + 1) * w["aging"]
            
        if "fairness" in self.ablations and self.ablations["fairness"]: F = 0
        else:
            j = task.group_id
            c_j = self.group_gpu_hours.get(j, 0)
            s_j = self.group_target_share[j]
            D_j = s_j - c_j
            F = math.tanh(D_j / Config.FAIRNESS_LAMBDA) * w["fairness"]
        return B + V + U + A + F

    def _allocate(self, task, gpus, current_time, duration):
        for g in gpus:
            g.busy_until = current_time + duration
            g.alloc_task = task.task_id
        task.start_time = current_time
        task.finish_time = current_time + duration
        task.alloc_gpus = [g.id for g in gpus]

    def schedule(self, wait_queue, current_time):
        # 1. 队列排序
        if self.strategy == "FCFS":
            wait_queue.sort(key=lambda t: t.arrival_time)
        elif self.strategy == "Static":
            wait_queue.sort(key=lambda t: (-t.priority, t.arrival_time))
        else:
            for t in wait_queue:
                t.dynamic_p = self.get_priority(t, current_time)
            wait_queue.sort(key=lambda t: -t.dynamic_p)
            
        scheduled = []
        
        # 2. 遍历队列，执行调度与回填
        for i, task in enumerate(wait_queue):
            avail_gpus = [g for g in self.gpus if g.busy_until <= current_time]
            
            # 如果资源足够，直接调度
            if len(avail_gpus) >= task.num_gpus:
                # 本文方法进行资源适配度评估
                if self.strategy == "PCBP" and not ("fitness" in self.ablations and self.ablations["fitness"]):
                    best_fit = -1; best_gpus = None
                    avail_sorted = sorted(avail_gpus, key=lambda g: (g.node_id, g.vram))
                    for idx in range(len(avail_sorted) - task.num_gpus + 1):
                        candidate = avail_sorted[idx:idx+task.num_gpus]
                        fit, _ = self.calculate_fitness(task, candidate)
                        if fit > best_fit:
                            best_fit = fit; best_gpus = candidate
                    if best_gpus:
                        self._allocate(task, best_gpus, current_time, task.duration)
                        scheduled.append(task)
                else:
                    # 基线策略或消融适配：直接取前N个
                    selected = avail_gpus[:task.num_gpus]
                    self._allocate(task, selected, current_time, task.duration)
                    scheduled.append(task)
            else:
                # 3. 队首任务资源不足！触发阻塞与回填逻辑
                if self.strategy == "FCFS":
                    # FCFS 严格排队，队首不满，后面全部等待
                    break
                
                # Static 和 PCBP 支持回填
                if not ("backfill" in self.ablations and self.ablations["backfill"]):
                    # 向后扫描寻找可回填任务
                    for bf_task in wait_queue[i+1:]:
                        if bf_task in scheduled: continue
                        
                        # 实时计算当前还剩余的空闲GPU
                        current_avail = [g for g in self.gpus if g.busy_until <= current_time and g.id not in [gid for t in scheduled for gid in t.alloc_gpus]]
                        
                        if len(current_avail) >= bf_task.num_gpus:
                            safe_to_run = False
                            
                            if self.strategy == "PCBP":
                                # 预测型保守回填：必须有检查点，或预测时间很短(小于0.5h)
                                if bf_task.has_ckpt or bf_task.predicted_duration <= 30:
                                    safe_to_run = True
                            else:
                                # 静态优先级的简单回填：只要有资源就塞
                                safe_to_run = True
                            
                            if safe_to_run:
                                selected = current_avail[:bf_task.num_gpus]
                                self._allocate(bf_task, selected, current_time, bf_task.duration)
                                scheduled.append(bf_task)
                
                # 队首资源不足触发回填后，停止继续向后调度，等待下一轮
                break
                    
        return scheduled

class Simulator:
    def __init__(self, strategy, ablations={}):
        self.strategy = strategy
        self.ablations = ablations
        self._load_env()
        self._load_tasks()
        self.gpus = []
        for node in self.nodes:
            for g in node["gpus"]:
                self.gpus.append(GPU(g["gpu_id"], g["cap"], g["vram"], node["node_id"]))
        
    def _load_env(self):
        with open("cluster_env.json", "r") as f:
            self.nodes = json.load(f)
            
    def _load_tasks(self):
        with open("workload_tasks.json", "r") as f:
            self.tasks = [Task(data) for data in json.load(f)]
            
    def run(self):
        scheduler = Scheduler(self.gpus, self.nodes, self.strategy, self.ablations)
        pending = sorted(self.tasks, key=lambda t: t.arrival_time)
        wait_queue = []
        running_queue = []
        finished = []
        
        current_time = 0
        while pending or wait_queue or running_queue:
            # 1. 释放已完成任务资源
            still_running = []
            for t in running_queue:
                if t.finish_time <= current_time:
                    finished.append(t)
                    j = t.group_id
                    scheduler.group_gpu_hours[j] += (t.finish_time - t.start_time) * len(t.alloc_gpus) / 3600.0
                    for gid in t.alloc_gpus:
                        self.gpus[gid].busy_until = 0
                        self.gpus[gid].alloc_task = None
                else:
                    still_running.append(t)
            running_queue = still_running
            
            # 2. 任务到达入队
            ready = [t for t in pending if t.arrival_time <= current_time]
            pending = [t for t in pending if t.arrival_time > current_time]
            wait_queue.extend(ready)
            
            # 3. 执行调度
            scheduled = scheduler.schedule(wait_queue, current_time)
            for t in scheduled:
                if t in wait_queue: wait_queue.remove(t)
                if t not in running_queue: running_queue.append(t)
                
            current_time += Config.TIME_STEP_MIN
            if current_time > Config.MAX_SIM_TIME_MIN: break
                
        return finished, current_time

    def evaluate(self, finished, end_time):
        total_wait = 0; total_duration = 0; wait_times = []
        deadline_total = 0; deadline_violation = 0
        high_pri_total = 0; high_pri_violation = 0
        gpu_busy_hours = 0; gpu_effective_hours = 0
        
        for t in finished:
            wait_h = (t.start_time - t.arrival_time) / 60
            dur_h = t.duration / 60
            total_wait += wait_h; total_duration += dur_h
            wait_times.append(wait_h)
            
            if t.deadline != float('inf'):
                deadline_total += 1
                if t.finish_time > t.deadline: deadline_violation += 1
                if t.priority == 1:
                    high_pri_total += 1
                    if t.finish_time > t.deadline: high_pri_violation += 1
                    
            n = len(t.alloc_gpus)
            if n == 0: continue
            
            s_n = 1.0 / (1 + 0.15 * (n - 1)) if n > 1 else 1.0
            node_set = set()
            for gid in t.alloc_gpus: node_set.add(self.gpus[gid].node_id)
            if len(node_set) > 1: s_n *= 0.8 ** (len(node_set) - 1)
            
            progress_coeff = 1.0
            avg_vram = sum(self.gpus[gid].vram for gid in t.alloc_gpus) / n
            avg_cap = sum(self.gpus[gid].cap for gid in t.alloc_gpus) / n
            if avg_vram > t.mem_per_gpu * 4: progress_coeff *= 0.8
            if t.mem_per_gpu > avg_vram * 0.7: progress_coeff *= 0.4
            if t.num_gpus > 2 and avg_cap < 1.0: progress_coeff *= 0.5
            
            gpu_busy_hours += dur_h * n
            gpu_effective_hours += dur_h * n * s_n * progress_coeff
            
        avg_wait = total_wait / len(finished) if finished else 0
        p95_wait = sorted(wait_times)[int(len(wait_times)*0.95)] if wait_times else 0
        total_cluster_gpu_hours = Config.TOTAL_GPUS * (end_time / 60.0)
        util_apparent = (gpu_busy_hours / total_cluster_gpu_hours) * 100 if total_cluster_gpu_hours > 0 else 0
        util_effective = (gpu_effective_hours / total_cluster_gpu_hours) * 100 if total_cluster_gpu_hours > 0 else 0
        viol_rate = (deadline_violation / deadline_total * 100) if deadline_total > 0 else 0
        high_val_comp = ((1 - high_pri_violation/high_pri_total) * 100) if high_pri_total > 0 else 100
        
        sum_w = sum(wait_times); sq_sum = sum(w**2 for w in wait_times)
        jain = (sum_w**2) / (len(wait_times) * sq_sum) if sq_sum > 0 else 1
        
        return {
            "Avg Wait (h)": f"{avg_wait:.1f}",
            "95p Wait (h)": f"{p95_wait:.1f}",
            "Util Apparent (%)": f"{util_apparent:.1f}",
            "Util Effective (%)": f"{util_effective:.1f}",
            "Viol Rate (%)": f"{viol_rate:.1f}",
            "High Val Comp (%)": f"{high_val_comp:.1f}",
            "Jain Index": f"{jain:.2f}"
        }