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

trace PPO 默认只在报告候选集 `C01 C03 C04 C05 C09` 中选择；可通过 `--strategies` 修改候选集。训练接口还支持先用 `expert_data` 和 `bc_epochs` 做行为克隆预训练，再进行 PPO 更新。

可使用 `resume_from` 或命令行 `--resume-from` 从已有 `ppo_model.pt` 继续训练，避免每次从随机初始化重新开始。

`src/experiments/trace_demonstrations.py` 可从多个真实窗口生成专家状态序列；环境终局奖励同时扣除平均完成时间，避免 PPO 只根据“是否完成”学习而忽略速度。

任务需求不再由随机四维权重直接生成。`src/models/task_demand.py` 根据截止期裕度、等待时间、优先级、依赖影响、失败代价和可行资源数量计算连续需求向量：`urgency`、`criticality`、`throughput_preference`、`cost_sensitivity`、`stability_requirement` 和 `resource_scarcity`。PPO 观测新增 `demands` 字段；紧急度和关键度是可解释的连续状态特征，不是人工固定标签。显式传入旧版 `objective_weights` 的任务仍保持兼容。

正式小规模评估使用 250000–260000 时间窗口、20 个完整作业和 2 台机器；PPO 训练 5 个 episode、每回合最多 300 步，完整回合为 194 步。该次模型评估结果保存在本地 `results/trace-formal/ppo-long/evaluation.json`，当前平均完成时间为 70.63，仍略慢于该窗口最优固定策略 C04/C09 的 68.86，因此只能作为当前阶段基线，不能写成 PPO 已经提升。

候选集修正后的 40 episode 结果保存在 `results/trace-formal/ppo-candidates/`；同一窗口上 PPO 平均完成时间为 68.86，与 C04/C09 持平，说明动作空间缩减有效消除了原先的 2.57% 劣化，但还需要独立窗口验证自适应策略是否能进一步超过固定策略。

加入终局完成时间奖励后的多窗口结果保存在 `results/trace-mixed/ppo-reward/`；两个窗口分别为 68.86 和 389.14。当前模型仍偏向 C04，尚未超过每个窗口的最优固定策略，因此后续应继续做状态特征和动态切换消融，不能把该结果写成 PPO 已取得加速。

从 `ppo-reward/ppo_model.pt` 继续训练 150 个 episode 后，模型仍主要选择 C04，两个窗口结果仍为 68.86 和 389.14。说明单纯增加训练轮数已经不能解决跨窗口策略切换问题。

随后将系统状态第 6 维从当前仿真时间改为未完成任务的总剩余工作量，并重新训练 100 个 episode；结果保存在 `results/trace-mixed/ppo-workload-state/`。独立窗口平均完成时间改善到 355.75，与最优固定策略持平；训练窗口为 69.45，较 C04/C09 的 68.86 慢约 0.85%。

网络进一步对任务嵌入使用平均池化和最大池化联合表示，结果保存在 `results/trace-mixed/ppo-rich-task-state/`；两个窗口结果仍为 69.45 和 355.75，说明当前瓶颈不是简单增加训练轮数或池化统计量，而是需要更细粒度的状态级策略切换机制。

细粒度切换已实现：trace PPO 默认每个动作最多分配一个任务，下一动作可重新选择策略；固定策略也在相同粒度下重跑。结果保存在 `results/trace-mixed/ppo-fine-grained/` 和 `results/trace-mixed/fine-grained-baselines.json`：PPO 在 250000–260000 窗口为 68.8587，在 260000–270000 窗口为 355.7500，均达到对应窗口最优固定策略，但暂未超过。

`src/experiments/trace_demonstrations.py` 进一步支持局部贪心专家标签：每个决策点重放历史并比较候选动作的即时回报。`results/trace-mixed/ppo-local-expert/` 的训练结果为 69.4457 和 355.7500，说明即时标签能保持泛化，但要超过固定策略还需要多步前瞻标签。

已加入 horizon=3 的多步前瞻专家标签；`results/trace-mixed/ppo-lookahead/` 的结果为 69.4457 和 355.7500。该实验没有超过固定策略，说明在当前两个窗口中候选策略差异不足以证明 PPO 的超越性，后续需要加入更丰富的负载变化和故障扰动场景。

四窗口训练结果保存在 `results/trace-four-windows/`。在 150000–160000、200000–210000、250000–260000、260000–270000 四个窗口上，PPO 分别达到 95.7192、84.0128、69.4457、355.7500；新增窗口的同粒度固定策略最优值分别也是 95.7192 和 84.0128。当前 PPO 已在四个窗口稳定达到最优固定策略，但尚未产生超越。

真实轨迹故障训练也已接入：`TraceSchedulingEnv` 支持 `fault_resource/fault_at/fault_duration`，故障结果保存在 `results/trace-fault/`。在 m_1 相对时间 50 故障 30 秒的设置下，PPO 在第一个窗口达到 55.9762，与最优固定策略持平；第二个窗口为 402.0137，低于 C01/C03/C05 的 359.64，说明故障自适应需要正常/故障混合训练和更多故障位置消融。

训练器现在支持 `fault_profiles` 轮换正常、早期故障和晚期故障。混合训练结果保存在 `results/trace-mixed-fault/`；对未参与训练的相对时间 50 故障，两个窗口 PPO 分别为 56.6190 和 359.6400，第二窗口已达到最优固定策略，第一窗口距离 55.9762 约 1.15%。

为避免 PPO 更新冲掉细粒度专家标签，训练器新增 `bc_epochs_per_update` 专家约束。合成动态场景对比保存在 `results/synthetic-dynamic/ppo-regularized/comparison.json`：seed=80 上 PPO 为 10.6667，最优固定策略为 11.6667，提升 8.57%；seed=9 提升 2.78%；seed=12 提升 2.17%。同时记录了 seed=27/73/43 的退化结果，说明提升并非所有场景都稳定，需要在报告中同时呈现均值和方差。

从该模型继续训练 500 个 episode 后，最终结果保存在 `results/synthetic-dynamic/ppo-continued/comparison.json`：seed=27、80、73、43、9、12 的 PPO 提升分别为 9.72%、8.57%、10.53%、2.00%、2.78%、2.17%，六个测试场景全部超过最优固定策略。该结论适用于合成动态混合任务集；Alibaba 真实 trace 上目前仍以达到最优固定策略为主，不能混用两类结论。

真实四窗口使用多步专家标签和每轮专家约束的复现实验保存在 `results/trace-four-windows/ppo-regularized/`；四个窗口仍为 95.7192、84.0128、69.4457、355.7500，均与同粒度最优固定策略持平。该结果表明当前 Alibaba 子集缺少足够的策略切换收益，不能通过继续增加训练轮数强行制造提升。
