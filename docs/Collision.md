可以，按你的思路，最合适的是把它做成一个**面向鲁棒性评测的数据生成器**，而不是直接照搬 STRIVE。
在 NAVSIM 里更自然的形式是：

**给定原始 scene → 选攻击车 → 选攻击点/时刻 → 用 IDM+MOBIL 生成可行交互轨迹 → 重新跑 ego planner → 保留那些显著拉低 EPDMS/造成危险但仍然物理合理的样本。**

这套做法和 NAVSIM 的接口是兼容的，因为 agent 侧仍然只需要输出 4 秒轨迹，评测仍然基于 4 秒、10Hz 的 `compute_trajectory()` 输出；NAVSIM v2 主分支还支持两阶段 pseudo closed-loop 和 reactive traffic agents，官方文档也明确提到 reactive traffic agents 采用 IDM。([GitHub][1])

下面我把你缺的“细节”补成一个能落地的方案。

---

## 1. 先定清楚你要优化什么

别一上来就“找最危险轨迹”，先把目标拆成三层：

### A. 攻击目标

你要让 ego 出现哪类失败：

* at-fault collision
* TTC 很差
* 进度骤降
* 红灯/逆行/出界
* 两阶段 EPDMS 明显下降

在 NAVSIM v2 里，EPDMS 由 `NC / DAC / DDC / TLC` 这些乘法 filter，加上 `TTC / EP / LK / HC / EC` 这些加权项组成；而 v2 还会把第一阶段和多个 follow-up scene 做两阶段聚合。([GitHub][2])

所以你的外层优化目标最好直接写成：

[
J_{\text{attack}} =
w_1(1-\text{NC}) +
w_2(1-\text{TTC}) +
w_3(1-\text{EP}) +
w_4(1-\text{LK}) +
w_5(1-\text{EPDMS})
]

更实用一点，可以分两档：

* **硬目标**：优先让 NC/TTC 变差
* **软目标**：再追求 EPDMS 降得更多

---

## 2. 攻击车怎么选：不要全量搜，先做候选筛选

你说“找到最适合的攻击车”，这里不要直接 brute-force 所有车。
建议先做一个 **candidate ranking**，只保留 top-k。

### 2.1 候选筛选条件

对每个背景车 (i)，在原始 scene 上算下面这些量：

[
S_i =
\alpha_1 \cdot \text{front_interaction}_i +
\alpha_2 \cdot \frac{1}{\min TTC_i + \epsilon} +
\alpha_3 \cdot \frac{1}{\min Dist_i + \epsilon} +
\alpha_4 \cdot \text{route_conflict}_i +
\alpha_5 \cdot \text{lane_change_feasible}_i
]

具体定义：

### (1) front_interaction

这个车是不是在 ego 前方、侧前方、交叉口冲突区。
后方远车通常不适合当攻击者。

### (2) min TTC / min Dist

基于原始未来或短时 rollout，算 ego 与该车 4 秒内的最小 TTC、最小距离。

### (3) route_conflict

看该车参考 lane sequence 和 ego route 是否存在：

* 同车道跟驰冲突
* merge 冲突
* unprotected turn 冲突
* 十字路口冲突

### (4) lane_change_feasible

MOBIL 能不能让它合理地并入 ego 路径附近：

* 左/右侧是否有候选 lane
* 目标 lane 是否连续
* 变道后前后车间距是否够
* 变道不至于让后车产生过大减速度

### 2.2 候选类别

你会发现最有效的攻击车通常就三类：

* **同车道前车**：急减速/诱导追尾或大幅减速
* **相邻车道侧车**：切入 ego 前方
* **交叉口冲突车**：抢行、晚让行、右转/左转卡位

建议每个场景最多留 `top 3~5` 个候选车，后面再做内层搜索。

---

## 3. 攻击点怎么选：空间点其实要转成“时空窗”

你说的“攻击点”不要只定义成地图上的一个点，最好定义成：

[
\mathcal{A} = (s^*, t^*, \text{mode})
]

其中：

* (s^*)：沿 ego route 的弧长位置
* (t^*)：预计冲突发生时刻
* `mode`：攻击模式

### 3.1 攻击点候选来源

从 ego 的 nominal trajectory 里找这些位置：

* stop line 前 5~15m
* merge 起点 / merge 中段
* 路口 conflict zone 中心
* lane narrowing / 障碍绕行点
* 计划换道落点附近
* 前车跟驰安全余量最小点

### 3.2 三种攻击模式

建议先只做这三种，够用了：

#### 模式 1：cut-in

背景车从相邻 lane 变到 ego 前方，利用 MOBIL 决策变道，用 IDM 控速。

适合：

* 双车道直路
* merge
* 路口出口

#### 模式 2：brake-check

同车道前车在 ego 前方进行较强减速，但不能夸张到脱离自然驾驶。

适合：

* 跟驰
* 红灯前
* 拥堵流

#### 模式 3：intersection seize

交叉车在 conflict zone 抢占时空窗口，逼 ego 急刹或碰撞。

适合：

* unprotected left
* 十字路口直行/右转冲突
* 环岛进入

### 3.3 攻击点评分

对某个候选攻击点 ((s^*, t^*, mode))，你可以定义：

[
Q = \beta_1 \cdot \text{ego_commitment}

* \beta_2 \cdot \text{traffic_plausibility}
* \beta_3 \cdot \text{collision_margin_reduction}
  ]

解释一下：

* `ego_commitment`：ego 是否已经做出不可逆决策，比如开始加速穿路口、开始并线
* `traffic_plausibility`：这个点上背景车做该动作是否合理
* `collision_margin_reduction`：实施后安全余量能缩小多少

攻击点不是越近越好，而是**ego 已经 committed，但背景车仍有可行动作空间**的时候最好。

---

## 4. 内层生成：用 IDM+MOBIL 不是“直接跑”，而是“参数化跑”

这一层最关键。

你不能只说“用 IDM+MOBIL”，要把它变成**可搜索参数**。

## 4.1 攻击车参数化

给每个候选攻击车定义参数向量：

[
\theta =
[v_0,\ T,\ a,\ b,\ s_0,\ \Delta,\ p_{\text{lc}},\ a_{\text{thr}},\ t_{\text{start}}]
]

其中：

* (v_0)：期望速度
* (T)：期望时距
* (a)：最大加速度
* (b)：舒适减速度
* (s_0)：最小间距
* (\Delta)：IDM 指数
* (p_{lc})：MOBIL politeness
* (a_{thr})：变道收益阈值
* (t_{start})：开始攻击的时刻

再加模式相关参数：

* cut-in：目标 lane、最晚变道时刻、最小接受 gap
* brake-check：最大减速度上限、持续时间
* intersection seize：进冲突区的期望到达时刻偏移

### 4.2 参数范围

为了保持自然性，建议把参数限制在“正常驾驶”范围，不要过拟合成奇怪驾驶：

* (T \in [0.8, 2.0]) s
* (a \in [0.8, 3.0]) m/s²
* (b \in [1.5, 4.5]) m/s²
* (s_0 \in [1, 5]) m
* politeness (p \in [0, 0.5])
* 变道收益阈值 (a_{thr} \in [0.05, 0.5]) m/s²

这样生成的数据更像“激进但合理”的驾驶，而不是明显伪造。

---

## 5. 不要只改一辆车的轨迹，要考虑局部一致性

这是很多人做攻击场景时最容易忽略的地方。

如果你只改攻击车，旁边车都保持日志真值，很容易出现：

* 攻击车切入了，但后车毫无反应
* 攻击车急刹了，但后方队列不跟随
* 路口抢行了，横向车流没变化

在 NAVSIM v2 里，reactive traffic agents 本来就是为了让背景交通随 ego 产生交互，官方说明里也提到 reactive agents 基于 IDM，并在两阶段评测中使用。([GitHub][1])

所以你最好这样做：

### 局部联动更新

如果攻击车被修改，就把它 20~40m 邻域内的车辆也做一次短 horizon 联动 rollout：

* 同车道后车：用 IDM 跟车
* 目标 lane 后车：用 IDM 响应 cut-in
* 冲突区相关车：根据优先级规则做轻微反应

这样生成的新 scene 更自然。

---

## 6. NAVSIM 下的完整生成流程

给你一个最实用的整体 pipeline。

### Step 0：读取 scene 和 cache

NAVSIM 的 `Scene` / `Frame` 是基本数据结构，OpenScene 是对 nuPlan 的紧凑重分发；评测前建议先生成 metric cache，因为地图访问和坐标变换预处理比较重。([GitHub][3])

### Step 1：跑 nominal ego

用你当前 planner 在原始 scene 上跑一次，得到 nominal 4 秒轨迹。

### Step 2：候选攻击车筛选

根据 TTC、最小距离、route conflict、lane-change feasibility 打分，取 top-k。

### Step 3：候选攻击点生成

围绕 nominal ego trajectory 的关键点生成 `(s*, t*, mode)`。

### Step 4：内层搜索攻击参数

对每个 `(vehicle, attack_point, mode)` 做参数搜索，生成修改后的背景交通。

搜索器你先别用梯度，直接用：

* grid search + prune
* beam search
* CEM
* CMA-ES

都可以。

### Step 5：重新评 ego

在修改后的 scene 上重新跑 ego planner，得到新轨迹。

### Step 6：计算攻击收益

算：

* min TTC
* min distance
* collision type
* EPDMS drop
* 是否保持自然性约束

### Step 7：保留样本

只保留同时满足：

* 危险性提升
* 自然性合格
* 地图合法
* 动力学合法
* 局部交通一致

的样本。

---

## 7. 目标函数怎么写最稳

我建议你用“危险性 - 不自然性”的形式。

[
\max_{\theta, i, \mathcal{A}}
\quad
R_{\text{risk}} - \lambda_1 R_{\text{implausible}} - \lambda_2 R_{\text{map_violation}} - \lambda_3 R_{\text{dyn_violation}}
]

其中：

### 风险项

[
R_{\text{risk}} =
\eta_1 (1-\text{NC}) +
\eta_2 (1-\text{TTC}) +
\eta_3 (1-\text{EPDMS}) +
\eta_4 \cdot \mathbb{1}[\text{ego emergency brake}]
]

### 不自然项

[
R_{\text{implausible}} =
c_1 |\Delta a| +
c_2 |\Delta jerk| +
c_3 |\Delta v_{\text{pref}}| +
c_4 \mathbb{1}[\text{unreasonable lane change}]
]

### 地图约束

* 出 drivable area 重罚
* 逆行重罚
* 穿实体边界重罚
* 红灯违法可选：如果你要自然驾驶数据，就罚；如果你要 stress test，可允许但单独标注

---

## 8. 攻击生成时，三类模式的具体实现细节

## 8.1 Cut-in 模式

### 什么时候触发

* 攻击车在相邻车道
* ego 预计 1~3 秒后到达 merge/cut-in 区域
* 目标 lane gap 勉强可接受

### 怎么做

1. 用 MOBIL 判断切入 ego lane 是否有收益
2. 如果收益不足，允许对参数轻微调节：

   * 降低 politeness
   * 降低 lane-change threshold
   * 提高 desired speed
3. 触发变道后，用 IDM 保持前向速度，控制其切入 ego 前方的时空位置

### 关键细节

别让它“切完立刻急刹”，那样太假。
先 cut-in，再按 IDM 自然减速或维持速度。

---

## 8.2 Brake-check 模式

### 什么时候触发

* 攻击车已在 ego 前方同车道
* ego 跟驰距离不算太大
* 前方存在合理减速理由，如轻微拥堵、停止线、前车减速

### 怎么做

1. 不直接插入一个硬刹车轨迹
2. 而是调 IDM 参数：

   * 降低 desired speed
   * 提高 headway sensitivity
   * 在 (t_{start}) 后抬高期望减速度响应
3. 让它呈现“稍强但仍自然”的减速

### 关键细节

加一个 jerk 上限，否则很容易一眼假。

---

## 8.3 Intersection seize 模式

### 什么时候触发

* ego 正在接近冲突区
* 攻击车有合法进入冲突区的 lane/path
* ego 当前策略对对向/横向车比较依赖预测

### 怎么做

1. 选一个冲突点 (P_c)
2. 估计 ego 到达 (P_c) 的 nominal 时间 (t_e)
3. 搜索攻击车参数，让它到达 (P_c) 的时间落在：
   [
   t_a \in [t_e - \delta_1, t_e + \delta_2]
   ]
4. 用 IDM 控速，必要时允许更激进的 gap acceptance

### 关键细节

这里不是追求绝对碰撞，而是追求**压缩 ego 的可恢复空间**。
很多时候让 ego 进入“急刹 + 低进度 + lane keeping 变差”的状态，比硬碰撞更稳定。

---

## 9. 新数据怎么存：建议存成“攻击元数据 + 重建参数”，不要只存轨迹

为了后续复现实验，建议每个生成样本除了新轨迹，还存一份 attack manifest：

```json
{
  "scene_id": "...",
  "ego_agent_name": "...",
  "attack_vehicle_id": 37,
  "attack_mode": "cut_in",
  "attack_point_s": 54.2,
  "attack_start_t": 1.6,
  "idm_params": {...},
  "mobil_params": {...},
  "neighbors_updated": [12, 15, 41],
  "nominal_metrics": {...},
  "attacked_metrics": {...},
  "risk_gain": ...,
  "plausibility_score": ...
}
```

这样你之后才能分析：

* 哪类攻击模式最有效
* 哪类 planner 最脆弱
* 哪些地图结构最容易出问题

---

## 10. 一个你可以直接实现的伪代码

```python
for scene in dataset:
    nominal_traj = ego_planner(scene)

    candidate_vehicles = rank_attack_vehicles(scene, nominal_traj, top_k=5)

    best_attack = None
    best_score = -inf

    for veh in candidate_vehicles:
        attack_points = generate_attack_points(scene, nominal_traj, veh)

        for attack_point in attack_points:
            for mode in feasible_modes(veh, attack_point):
                for theta in search_params(mode):
                    attacked_scene = rollout_local_traffic(
                        scene=scene,
                        attack_vehicle=veh,
                        attack_point=attack_point,
                        mode=mode,
                        params=theta,
                        model="IDM+MOBIL",
                        local_consistency=True,
                    )

                    if not valid_scene(attacked_scene):
                        continue

                    attacked_traj = ego_planner(attacked_scene)
                    metrics = evaluate(attacked_scene, attacked_traj)

                    score = attack_objective(metrics, attacked_scene)

                    if score > best_score:
                        best_score = score
                        best_attack = {
                            "scene": attacked_scene,
                            "vehicle": veh,
                            "point": attack_point,
                            "mode": mode,
                            "params": theta,
                            "metrics": metrics,
                        }

    if best_attack and pass_filters(best_attack):
        save(best_attack)
```

---

## 11. 你最容易踩的坑

### 坑 1：只看碰撞，不看 EPDMS

这样会生成很多“很危险但评测不敏感”的样本。
要把 EPDMS drop 也纳入目标。

### 坑 2：攻击车选得太远

远车能优化的空间大，但实际对 ego 没什么作用。
优先同车道前车、侧前车、冲突区来车。

### 坑 3：MOBIL 只做单车，不更新周围车

会导致生成的数据很假。

### 坑 4：攻击点只按地图点，不按时序

真正决定效果的是“ego 何时 committed”，不是几何位置本身。

### 坑 5：保留所有成功攻击样本

最后数据会偏到极端分布。
建议按危险度和自然度做分桶采样。

---

## 12. 我建议你的第一版配置

先别搞太复杂，直接上这个版本：

* 候选攻击车：top 3
* 攻击模式：`cut-in`、`brake-check`
* 搜索器：CEM 或小规模 grid search
* horizon：4 秒
* 局部联动车：攻击车前后各 1 辆 + 目标 lane 后车 1 辆
* 目标：先最小化 TTC，再最小化 EPDMS
* 保留条件：

  * 不出界
  * jerk / accel 不超阈值
  * 至少有一项核心风险指标变差
  * scene 仍可被 NAVSIM evaluator 跑通

这是最容易先跑出结果的版本。

如果你要，我下一条可以直接继续给你两样东西里的一个：
**A. “IDM+MOBIL 攻击生成器”的模块化代码骨架**，或者 **B. “攻击车选择 + 攻击点选择”的具体打分公式和阈值表**。

[1]: https://github.com/autonomousvision/navsim/blob/main/README.md "navsim/README.md at main · autonomousvision/navsim · GitHub"
[2]: https://github.com/autonomousvision/navsim/blob/main/docs/metrics.md "navsim/docs/metrics.md at main · autonomousvision/navsim · GitHub"
[3]: https://github.com/autonomousvision/navsim/blob/main/docs/cache.md "navsim/docs/cache.md at main · autonomousvision/navsim · GitHub"
