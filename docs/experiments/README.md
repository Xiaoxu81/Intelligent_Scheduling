# 实验数据说明

当前实验数据全部来自本地可复现仿真，不包含真实企业生产数据，也不是从互联网下载的数据。

真实数据适配器已按 Alibaba Cluster Trace v2018 的公开 schema 实现。原始数据需要用户从 [Alibaba Cluster Trace Program](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2018) 按其发布页面说明获取；原始压缩包很大，不应提交到本仓库。

## 数据生成链路

1. `src/experiments/scenarios.py` 使用固定随机种子生成任务、资源、依赖、截止期和任务目标权重。
2. `src/environment/simulation.py` 使用 SimPy 推进任务到达、调度、执行、抢占和故障事件。
3. `src/experiments/evaluator.py` 在相同场景上运行多个候选策略并采集指标。
4. `src/experiments/run_baselines.py` 和 `src/experiments/run_ppo.py` 保存实验配置、随机种子和训练/评价结果。

每个结果文件的 `metadata.data_source` 应为 `synthetic_simulation`。在报告中，这些结果只能用于说明方法原型和仿真验证，不能表述为真实产业链数据结论。

当使用 `src/experiments/alibaba_trace.py` 读取真实 trace 后，结果元数据应改为 `alibaba_cluster_trace_v2018`，并在论文中说明它是生产集群调度数据，不是产业链订单数据。

## 当前仿真数据规模

已运行的基线 smoke 实验使用 3 个随机种子，每个种子重复 3 次，每个场景包含 8 个任务和 2 个资源。该规模用于验证实验流水线，正式报告还需要扩大样本量并补充故障、消融和泛化场景。

## 运行示例

```powershell
$env:PYTHONPATH = (Get-Location).Path
python -m src.experiments.run_baselines --seed 1 --tasks 8 --resources 2 --repeats 3 --fault-profile single --output results/baselines-fault/seed1
```

## 真实 trace 子集实验

下载并解压 Alibaba Cluster Trace v2018 后，只把本地文件路径传给入口；原始数据不会复制进仓库：

```powershell
$env:PYTHONPATH = (Get-Location).Path
python -m src.experiments.run_trace_baselines `
  --machine-meta D:\data\machine_meta.csv `
  --batch-task D:\data\batch_task.csv `
  --start-time 150000 `
  --end-time 160000 `
  --limit-jobs 20 `
  --limit-resources 10 `
  --strategies C01 C03 C04 C05 C09 `
  --output results/trace-baselines/subset1
```

该入口会生成 `result.json`、`strategies.json` 和 `tasks.csv`，并在 metadata 中记录数据来源、输入路径、时间窗口、完整作业筛选、样本限制和策略列表。默认会按 `job_name/task_name` 合并同一任务的多个实例，避免 DAG 节点被重复覆盖。

真实 trace 没有订单截止期字段，因此 `deadline_satisfaction_rate` 不能作为该数据集上的有效结论；应重点报告完成时间、吞吐量、资源利用率和故障恢复指标。真实 trace 的 `batch_task` 是集群批任务数据，不能直接当作供应链订单数据；论文中应将它作为调度方法的生产集群验证数据，并与仿真数据分开报告。

## 真实 trace PPO smoke

```powershell
python -m src.experiments.run_trace_ppo `
  --machine-meta D:\data\machine_meta.csv `
  --batch-task D:\data\batch_task.csv `
  --start-time 150000 `
  --end-time 160000 `
  --limit-jobs 20 `
  --limit-resources 2 `
  --episodes 3 `
  --max-steps 50 `
  --k-epochs 1 `
  --output results/trace-ppo/smoke
```

该命令用于验证真实 `S/T/R/wT` 状态能够进入 PPO 训练流程。正式实验需要增加 episode、设置独立验证窗口，并与固定策略在相同 trace 窗口上比较。

正式小规模评估使用 250000–260000 时间窗口、20 个完整作业和 2 台机器；PPO 训练 5 个 episode、每回合最多 300 步，完整回合为 194 步。该次模型评估结果保存在本地 `results/trace-formal/ppo-long/evaluation.json`，当前平均完成时间为 70.63，仍略慢于该窗口最优固定策略 C04/C09 的 68.86，因此只能作为当前阶段基线，不能写成 PPO 已经提升。
