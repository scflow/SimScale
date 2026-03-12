可以，**如果只以 NAVSIM 落地**，我建议你把“闯红灯并及时修正”做成一套**两阶段、事件条件化的伪仿真数据工厂**，而不是简单“改灯色+强行刹车”。这样最贴合 NAVSIM 的范式：规划器输出的是 **4 秒、10 Hz** 的未来轨迹；NAVSIM v2 本身也采用 **两阶段 pseudo closed-loop aggregation**，并支持**反应式 traffic agents**，但 **ego 仍然是单次提交一条 4 秒计划，不接收仿真中的环境回流**。这意味着你的方案要把“修正”预先编码进 ego 的 4 秒轨迹里，而不是指望 ego 在仿真中二次重规划。([GitHub][1])

## 总体思路

把每个样本都拆成：

**源场景**
从 OpenScene / NAVSIM 的原始 scene 里选一个受信号灯控制的交叉口片段。NAVSIM 数据以 `Scene -> Frame` 组织，OpenScene 是 nuPlan 的紧凑重分发版本，保留了 2Hz 的相关传感器和标注。([GitHub][2])

**阶段 A：违规边界状态构造**
不是直接规定“到 t+h 开始刹车”，而是让 ego 在未来 4 秒内的某个关键时刻先到达一个可控的**红灯违规边界状态**。

**阶段 B：恢复轨迹生成**
从这个边界状态继续完成同一条 4 秒轨迹的后半段，形成：

* 越线后停住
* 侵入斑马线后停住
* 已不宜停车时平稳通过
* 犹豫后停或犹豫后过

因为 NAVSIM 的 ego 是 nonreactive、single-plan，所以这两段最好在离线生成时一次性拼成完整 4 秒轨迹。([GitHub][3])

---

## 一、先定清楚 NAVSIM 里你要做什么，不做什么

这套方案里，NAVSIM 适合做的是：

1. **批量生成反事实样本**
2. **用统一 4 秒轨迹格式产出监督**
3. **用 NAVSIM 指标做初筛和回归测试**

不适合做的是：

* 真正闭环的 ego 二次决策
* 高保真多轮交互恢复
* “ego 看见他车变了，于是重新规划”的在线过程

因为 NAVSIM v2 虽然支持 reactive traffic agents，但文档明确写了：**周围车辆可以对 ego 响应，ego 自身仍是 nonreactive，环境更新不会回流进 planner；ego 必须提交一条覆盖整个仿真时域的单一计划。**([GitHub][3])

所以你这里的关键设计原则是：

> **ego 的“闯红灯 + 修正”必须是一条预先生成好的 4 秒轨迹；他车的反应由 traffic policy 承担。**

---

## 二、场景类型怎么选

建议只做 4 类，先小而稳：

### 1. 绿灯/黄灯边界 through → 改成红灯轻度误闯后停住

源样本：

* 原本能正常通过路口的 scene

目标：

* ego 在红灯后刚越停止线或进入斑马线浅层，然后停住

### 2. 黄转红边界 through → 中度误闯后停住

源样本：

* 原本黄灯边界通过

目标：

* ego 在未来 1–2 秒内形成更自然的“晚制动—越线—止损”

### 3. 深度侵入 → 平稳通过

源样本：

* 原本速度较高、接近停止线较近的 through scene

目标：

* 让 ego 到达“已不适合停”的边界状态，再平稳通过，而不是强行急刹

### 4. 红灯停车 → 误启动后短距离停住

源样本：

* 原本红灯等待的 scene

目标：

* ego 错误起步 0.5–2 m，然后意识到红灯并停住

这 4 类刚好覆盖：

* 接近路口时误闯
* 红灯静止误启动
* stop-recovery
* go-recovery

---

## 三、自车怎么设计

### 1. 自车不是“动作集采样”，而是“边界状态采样”

建议你定义一个 `z_violate`，表示 ego 在未来某个关键帧的违规状态：

* `s_rel_stopline`：相对停止线位置
  例如：`[-1.0m, +0.5m, +2.0m, +5.0m]`
* `zone_type`：线前 / 越停止线 / 斑马线 / 冲突区入口
* `v`：该时刻速度
* `a`：该时刻加速度
* `red_age`：红灯已持续多久
* `stoppability`：还能舒适停 / 只能强停 / 不宜停车

然后每个样本不是“采一个动作”，而是“采一个目标边界状态，再生成满足它的 4 秒轨迹”。

这样比直接改灯色自然得多。

---

### 2. 自车轨迹生成建议用“两段式纵向设计 + 原车道横向保持”

NAVSIM 评估的是未来 4 秒、10Hz 轨迹，所以最实用的是：

#### 横向

尽量**不改横向拓扑**：

* 沿原 route 和 lane centerline
* 只允许小横摆、小偏移
* 不做明显绕行

因为你的任务核心是红灯规则违例与纵向修正，不是避障绕行。

#### 纵向

把纵向分成两段：

**阶段 A：违规形成段**

* 通过调整相位、反应延迟、制动延迟，把 ego 推到 `z_violate`

**阶段 B：恢复段**

* 从 `z_violate` 开始继续完成 stop 或 go 的恢复

可以直接在 Frenet 坐标里做：

* `s(t)` 纵向弧长
* `d(t)` 近似保持 0
* 再映射回 `(x, y, yaw)`

这样生成最稳。

---

### 3. 自车的 4 类恢复模板

建议就做这 4 个模板：

#### A. `light_violate_stopline_stop`

* 红灯后轻度越线
* 峰值减速度中等
* 最终停在停止线后 0.5–2 m

#### B. `light_violate_crosswalk_stop`

* 红灯后侵入斑马线
* 中到强制动
* 最终停在斑马线内，但不进入冲突区深处

#### C. `light_violate_commit_go`

* 已不宜停车
* 维持可接受舒适度平稳通过
* 不允许停在路口中央

#### D. `redlight_false_start_stop`

* 红灯静止误启动
* 低速短距离前移
* 迅速刹停

你后续所有合成都围绕这 4 个模板展开。

---

### 4. 自车的参数怎么采样

建议控制这些参数，而不是采动作库：

* `phase_shift`：把原灯相位前移多少
* `reaction_delay`：对红灯变化的感知/响应延迟
* `brake_delay`：开始制动的延迟
* `a_brake_peak`：峰值减速度
* `j_max`：jerk 上限
* `target_intrusion_depth`：目标侵入深度
* `t_boundary`：到达违规边界状态的时间，一般设在未来 0.8–2.0 s 内
* `recovery_type`：stop / go

最关键的一点是：

**不要对所有样本都固定“1.5 秒后刹车”**。
而是固定“1.5 秒时落到某类边界状态”，再由可停性判定恢复方式。

---

### 5. 自车可停性判定

在生成恢复段前，先用一个简单判定：

* 若从 `z_violate` 到“安全停止位置”所需减速度 ≤ 舒适/可接受阈值
  → `stop-recovery`
* 若需要过大减速度，或会停在冲突区中央
  → `go-recovery`

这样可以避免模型学到“只要闯红灯就急停”的错误偏好。

---

## 四、他车怎么设计

这部分在 NAVSIM 里要分清楚：**车**和**非车**不一样。

官方 traffic agent 文档里写得很清楚：

* 可选 policy 包括 `Log-Replay`、`Constant-Velocity`、`IDM`
* **IDM 是 reactive 的，适用于车辆**
* **行人、静态物体和其他非车辆 agent 仍然跟随日志回放**。([GitHub][3])

所以你最好这样设计。

### 1. 机动车：主推 IDM reactive policy

对周围车辆，优先用 `IDM`：

* 前车会根据 ego 的闯入和减速调整跟驰
* 横向直行车在近冲突时也会体现更合理的纵向反应
* 比 log replay 更适合做“ego 误闯后，他车怎么变”

这和 NAVSIM 官方的 reactive traffic policy 方向一致。([GitHub][3])

### 2. 行人、非机动车、静态体：先 log replay

因为文档里写明非车辆 agent 仍跟随日志。([GitHub][3])

所以第一版建议：

* 行人继续沿日志
* 非机动车若被数据结构当作非车辆，也先 replay
* 静态障碍物不改

这样虽然不完美，但实现最稳，也符合 NAVSIM 现成能力边界。

### 3. 后车

后车是最容易被忽视、但对“误闯后刹停”最重要的对象。

建议单独加一个后车风险检查：

* ego stop-recovery 时，后车在 IDM 下是否会出现极低 TTC
* 若追尾风险过高，这个样本改为：

  * 降低 ego 侵入深度
  * 放缓刹车
  * 或直接改成 `go-recovery`

### 4. 横向直行车

对闯红灯场景最关键的不是“前方没车”，而是**横向冲突车**。

建议把样本筛选重点放在：

* 有横向交通流的 scene
* ego 闯入后，横向车通过 IDM 或 replay 是否形成潜在冲突
* 这样生成的数据才真有训练价值

---

## 五、推荐的 traffic policy 组合

如果你是做数据生成，不是做 leaderboard 复现，我建议：

### 方案 A：训练集生成

* 车辆：`IDM reactive`
* 行人/其他非车辆：`log replay`

适合生成“有止损意义”的样本。

### 方案 B：对照组

* 全部 `log replay`

适合看你的反事实构造是不是本身就不合理。

### 方案 C：调试组

* 车辆：`constant velocity`

只用于 debug，不建议用于最终数据，因为太假。官方文档也把 constant-velocity traffic agents 归为 debugging only。([GitHub][3])

---

## 六、如何从源场景构造样本

### 1. 场景筛选

从 `Scene / Frame` 里筛受信号灯控制的交叉口样本，保留：

* ego route 明确
* 未来 4 秒内临近停止线
* 存在交通灯状态
* 周边有一定交互对象

NAVSIM 场景数据和 metric cache 都是围绕 scene/frame 和地图局部化组织的，适合先用缓存筛场景，再做伪仿真。([GitHub][2])

### 2. 源类型分组

给每个源样本打 `source_type`：

* `green_through`
* `yellow_borderline_through`
* `red_wait`
* `red_follow_start`

### 3. 反事实扰动

根据 `source_type` 采不同扰动：

对于 `green_through`：

* 前移红灯相位
* 增加 `reaction_delay`
* 保留原大体路径

对于 `yellow_borderline_through`：

* 进一步提前红灯
* 增大 `brake_delay`

对于 `red_wait`：

* 加一个误启动模板
* 小幅前推再停

### 4. 生成 ego 4 秒轨迹

输出符合 NAVSIM agent 接口的未来轨迹：

* 本地坐标系下的 `x, y, heading`
* `TrajectorySampling`
* 4 秒 horizon，可按 10Hz 或较低频率输出再插值

这正好符合 NAVSIM 的 `compute_trajectory()` 输入输出约定。([GitHub][1])

---

## 七、怎么筛掉坏样本

NAVSIM v2 的 EPDMS 里，跟你这个任务最相关的是：

* `TLC`：Traffic Light Compliance
* `NC`：No at-fault Collisions
* `TTC`
* `DAC`
* `HC / EC`
* `LK`
  其中 v2 明确加强了舒适性、交通灯合规和驾驶方向合规的惩罚。([GitHub][4])

你做伪数据时，不要直接照搬 leaderboard 打分，而是把这些指标当过滤器：

### 必过项

* 无碰撞或不归责碰撞
* 不离开可行驶区域太多
* 不出现方向异常
* 轨迹连续

### 红灯任务专用过滤

* 必须发生目标类型的违规
* 必须有明确恢复行为
* 不能停在冲突区中央
* 不能有明显不物理的急刹 / jerk 峰值
* 若是 `go-recovery`，必须完整通过

### 舒适过滤

因为 NAVSIM v2 对 comfort 更敏感，建议你主动限制：

* 峰值减速度
* 峰值 jerk
* 相邻帧动态状态跳变
  这样更容易得到“可训练”的样本，也更不容易被 EC/HC 打崩。([GitHub][4])

---

## 八、建议的数据标注格式

每个样本建议包含：

### 样本级标签

* `scenario_type`
* `source_type`
* `recovery_type`
* `violation_level`
* `is_complete_event`

### 事件级标签

* `t_red`
* `t_cross_stopline`
* `t_enter_crosswalk`
* `t_enter_conflict_zone`
* `t_recovery_start`
* `t_recovery_end`
* `max_intrusion_distance`

### 序列级标签

未来 4 秒、每 0.1 秒一帧：

* `zone_state`
* `light_state`
* `behavior_state`
* `risk_state`

### 轨迹

* `traj[40, 3] = (x, y, yaw)`
* 可选再存 `speed, accel`

这样你既能训：

* 序列分类
* 轨迹预测
* 恢复策略选择
* 反事实规划评分

---

## 九、一个最实用的 NAVSIM-only pipeline

### Stage 0：缓存与筛场景

先利用 NAVSIM 的 metric cache / scene 数据，把候选交叉口片段筛出来。([GitHub][2])

### Stage 1：生成 ego 候选

对每个源样本生成若干 `z_violate`：

* 刚越线
* 侵入斑马线
* 冲突区入口
* 红灯误启动

### Stage 2：生成恢复段

对每个 `z_violate`：

* 计算可停性
* 生成 stop 或 go 的恢复轨迹
* 拼成完整 4 秒 ego trajectory

### Stage 3：跑 NAVSIM pseudo-sim

用：

* 车辆 `IDM reactive`
* 非车辆 `log replay`

### Stage 4：指标过滤

检查：

* TTC
* collision
* drivable area
* comfort
* TLC 相关
* 你的专用事件标签是否成立

### Stage 5：写回训练集

保存：

* scene token / frame id
* 修改后的交通灯配置
* ego 新轨迹
* traffic policy
* 标签

---

## 十、给你一版直接能落地的自车/他车默认配置

### 自车默认配置

* horizon：4.0 s
* frequency：10 Hz
* 横向：跟随原 lane centerline
* 纵向：两段式生成
* 模板比例：

  * 轻度越线停住：40%
  * 斑马线停住：30%
  * 深度侵入通过：20%
  * 红灯误启动：10%

### 他车默认配置

* 机动车：IDM reactive
* 行人：log replay
* 非机动车：先 log replay
* 静态障碍物：原样

### 样本保留规则

* 有明确信号灯状态
* 有停止线/斑马线几何
* ego 未来 4 秒内接近或处于路口
* 至少存在一个相关交互体：横向车、后车、行人三者之一

---

## 十一、最重要的一个提醒

在 NAVSIM 里，**不要把“及时修正”理解成“闯了就刹车”**。
更合理的定义是：

> 在 ego 已经接近或发生红灯违规后，在单次提交的 4 秒轨迹内，选择一个风险更低、动力学更可行、对交通参与者更一致的恢复方案。

所以你的数据里必须同时保留：

* `stop-recovery`
* `go-recovery`

否则模型会系统性学偏。

---

如果你愿意，我下一步可以直接给你补一版更工程化的内容：**NAVSIM 场景生成伪代码 + 自车/他车参数配置清单**。

[1]: https://github.com/autonomousvision/navsim/blob/main/docs/agents.md "navsim/docs/agents.md at main · autonomousvision/navsim · GitHub"
[2]: https://github.com/autonomousvision/navsim/blob/main/docs/cache.md?utm_source=chatgpt.com "navsim/docs/cache.md at main · autonomousvision/navsim · GitHub"
[3]: https://github.com/autonomousvision/navsim/blob/main/docs/traffic_agents.md "navsim/docs/traffic_agents.md at main · autonomousvision/navsim · GitHub"
[4]: https://github.com/autonomousvision/navsim/blob/main/docs/metrics.md "navsim/docs/metrics.md at main · autonomousvision/navsim · GitHub"

可以。下面我直接给你一版 **NAVSIM-only 的工程化方案**，重点包括：

1. **整体数据流**
2. **自车设计**
3. **他车设计**
4. **场景筛选与标注**
5. **伪代码**
6. **推荐参数表**

我会尽量贴合 NAVSIM 现有接口和评测范式来设计。NAVSIM 的 agent 接口要求输出未来轨迹 `Trajectory`；PDM/EPDM 评测的标准时域是 **4 秒、10 Hz**。同时，NAVSIM v2 支持 **reactive traffic agents**，但 **ego 自身仍然是 nonreactive**，也就是 ego 需要一次性提交整段未来计划，环境变化不会再回流给 ego 重新规划。OpenScene 是 nuPlan 的紧凑重分发版本，保留了 **2 Hz** 的相关标注和传感器数据。([GitHub][1])

---

# 1. 目标重述：在 NAVSIM 里你到底要生成什么

你的目标不是“在线仿真里 ego 先闯红灯，再下一拍重新决策”。在 NAVSIM 里，更现实的做法是：

> **离线生成一条完整的 4 秒 ego 轨迹**，这条轨迹内部已经包含
> “接近路口 → 形成红灯违规边界状态 → 采取恢复动作（停下或通过）”
> 这整个过程；然后把它放进 NAVSIM 的 pseudo-simulation 里，让他车对它反应，并用 NAVSIM 的 metric 做过滤。([GitHub][1])

所以整套系统应该叫：

**Red-Light Counterfactual Generator for NAVSIM**

输入是一个原始 `Scene / Frame`；输出是：

* 一条新的 ego 未来 4 秒轨迹
* 一套事件标签
* 一个“是否通过 NAVSIM 过滤”的标志

---

# 2. 整体数据流

建议做成 6 个模块：

### 模块 A：Scene 筛选器

从 OpenScene / NAVSIM 里挑出：

* 受信号灯控制的交叉口
* ego 在未来 4 秒内会接近停止线
* 地图里能拿到停止线、斑马线、车道中心线
* 周围有一定交互对象

NAVSIM 数据以 `Scene` 和 `Frame` 组织，官方也建议用 `SceneLoader` 直接读取 scene 级数据。([GitHub][2])

### 模块 B：源场景分类器

把候选场景分成 4 类：

* `green_through`
* `yellow_borderline_through`
* `red_wait`
* `red_follow_start`

这是工程层面的设计，不是 NAVSIM 官方定义；它是为了让后续扰动更稳。这个分法是我的建议。([GitHub][2])

### 模块 C：违规边界状态采样器

给 ego 设定一个未来某时刻的目标边界状态 `z_violate`：

* 相对停止线位置
* 当时速度
* 红灯已持续时间
* 侵入区域类型
* 是否仍可安全停车

这个 `z_violate` 不是 NAVSIM 原生对象，而是建议你自己定义的中间层数据结构。它能把“改灯色然后刹车”升级成“先到达某个违规边界，再决定如何恢复”。这是我基于你任务目标给出的设计。

### 模块 D：ego 两段式轨迹生成器

在 Frenet 或 route-aligned 坐标中生成：

* 第一段：从当前状态到 `z_violate`
* 第二段：从 `z_violate` 到最终恢复状态

最后映射回 NAVSIM 所需的 `(x, y, heading)` 轨迹。NAVSIM agent 输出格式就是 BEV pose 序列加 `TrajectorySampling`。([GitHub][1])

### 模块 E：traffic policy 配置器

给他车分配 policy：

* 机动车：`IDM reactive`
* 行人 / 非车辆：`log replay`
* 调试时才用 `constant velocity`

这和 NAVSIM v2 traffic-agents 的能力边界一致：车辆可做 reactive policy，ego 仍 nonreactive。([GitHub][3])

### 模块 F：metric + rule 过滤器

用 NAVSIM 的评测指标和你自定义的规则过滤样本：

* TTC
* at-fault collision
* drivable area
* lane keeping
* comfort / extended comfort
* traffic light compliance
* 以及你自定义的“恢复是否成立”规则

NAVSIM v2 的 metrics 明确强调 lane keeping、history comfort、extended comfort，以及驾驶方向/红灯合规等惩罚项。([GitHub][4])

---

# 3. 自车怎么设计

## 3.1 自车状态表示

建议在内部用如下状态：

[
x_t = (s_t, d_t, v_t, a_t, \psi_t)
]

其中：

* `s_t`：沿 route centerline 的弧长
* `d_t`：相对中心线横向偏移
* `v_t`：速度
* `a_t`：纵向加速度
* `psi_t`：航向

对你的任务来说，**横向基本保持不变**，核心变化在纵向。这个不是 NAVSIM 的硬性要求，而是因为你做的是“闯红灯并修正”，不是通用避障绕行。它也更容易满足 lane-keeping 指标。NAVSIM v2 对 lane keeping 和 comfort 都更敏感。([GitHub][4])

## 3.2 违规边界状态 `z_violate`

建议定义成：

```text
z_violate = {
  t_boundary,              # 未来第几秒达到边界状态
  s_rel_stopline,          # 相对停止线距离（m）
  zone_type,               # before_line / stopline / crosswalk / conflict_entry
  v_boundary,              # 边界速度
  a_boundary,              # 边界加速度
  red_age,                 # 红灯已持续时间
  stoppability             # can_stop / hard_stop / committed_go
}
```

推荐采样范围：

* `t_boundary`: 0.8 – 2.0 s
* `s_rel_stopline`: -1.0 m ~ +6.0 m
* `v_boundary`: 0.5 – 8.0 m/s
* `red_age`: 0.1 – 1.0 s

这组范围不是官方给的，是适合你任务的经验默认值；它的目的是把样本限制在“误闯后可修正”的区间，而不是极端恶意闯灯。

## 3.3 自车 4 个模板

建议保留这 4 类：

### 模板 A：`light_violate_stopline_stop`

* 红灯后刚越停止线
* 在停止线后短距离停住
* 用于“轻度违规 + 止损”

### 模板 B：`light_violate_crosswalk_stop`

* 红灯后进入斑马线浅中部
* 强一点的制动
* 最终不进入冲突区深处

### 模板 C：`light_violate_commit_go`

* 已接近冲突区或不适合停
* 平稳通过，而不是急停在中央

### 模板 D：`redlight_false_start_stop`

* 红灯静止误启动
* 短距离前移后立刻停住

这是任务分解建议，不是 NAVSIM 自带类型。它的优点是覆盖全面，同时每类都能映射成 4 秒轨迹监督。

---

# 4. 自车轨迹生成方法

## 4.1 第一段：从当前状态到违规边界状态

给定源场景和 `z_violate`，你生成一段 `0 ~ t_boundary` 的轨迹。

做法建议很简单：

* 横向：沿 route centerline，`d_t ≈ 0`
* 纵向：用五次多项式或分段 jerk-limited profile 生成 `s(t)`

约束：

* 起点状态匹配原 scene 当前 ego
* 终点状态匹配 `z_violate`
* 不允许倒车
* 不允许过大横摆
* 速度、加速度连续

这里你可以把原始“绿灯通过”改成“红灯误闯”的具体原因，编码成几个扰动参数：

* `phase_shift`
* `reaction_delay`
* `brake_delay`

这样比直接改灯色更自然。NAVSIM 本身不替你做这个逻辑，这是你伪数据生成器要做的。([GitHub][1])

## 4.2 第二段：从违规边界状态到恢复终态

### stop-recovery

满足：

* 所需减速度不超过阈值
* 停车点不落在冲突区中央
* 后车追尾风险可接受

则生成：

* 单调减速
* `a_min` 受限
* `jerk` 受限
* 最终 `v≈0`

### go-recovery

若从 `z_violate` 出发急刹不合适，则：

* 保持当前 lane
* 以舒适可接受的减速度/匀速通过
* 穿过冲突区，不在中间停

这个 stop/go 分流，和 NAVSIM 的 single-plan 机制是相容的：你离线先算好整条 4 秒轨迹，再提交给 simulator。([GitHub][5])

---

# 5. 他车怎么设计

## 5.1 车辆

第一版建议：

* **机动车全部用 IDM reactive**

原因：

* 你的任务里最关键的是 ego 闯灯后，后车、横向车会不会合理响应
* NAVSIM v2 就是为 surrounding vehicles 的 reaction 加了 support
* 这比 log replay 更适合“ego 偏离原日志行为”的伪数据场景

官方 issue 和文档都明确提到，v2 的 surrounding vehicles 可以对 ego 有意义地反应，从而避免一些 ego 并非过错方的碰撞。([GitHub][3])

## 5.2 行人 / 非车辆

第一版保持：

* **log replay**

因为官方 traffic-agents 设定里，反应式支持主要面向 surrounding vehicles；非车辆 agent 保持日志回放更稳。你这个任务最先解决的也应当是车-车交互，而不是立刻做复杂 pedestrian reaction。([GitHub][5])

## 5.3 静态物体

* 不改

## 5.4 三个必须重点检查的他车关系

### 后车

对 stop-recovery 样本，必须检查：

* 后车是否因 ego 急刹而变成极低 TTC
* 若高风险，则：

  * 减小侵入深度
  * 放缓刹车
  * 或改成 go-recovery

### 横向直行车

对闯红灯任务最关键的不是前车，而是横向冲突车。

* 若 ego 误闯后横向车完全没反应，样本可能太假
* 若横向车反应后仍形成高概率 at-fault 碰撞，样本应丢弃

### 前车

对 `redlight_false_start_stop` 类样本，前车反而不一定关键；你原先说“红灯前方没车就让 ego 闯”的想法太弱，真正更关键的是横向和后向关系。

---

# 6. 推荐的 traffic policy 组合

## 训练数据生成默认

* vehicles: `IDM reactive`
* pedestrian / bicycle / others: `log replay`

## 对照实验

* all agents: `log replay`

看你的伪样本是不是即使不引入 reactive traffic，也已经明显不自然。

## 调试实验

* vehicles: `constant velocity`

只用于 debug。constant velocity 适合验证几何和轨迹逻辑，不适合作为最终数据策略。([GitHub][5])

---

# 7. 场景筛选规则

建议先筛 scene，再做生成。筛选条件：

### 必须满足

* ego route 前方存在受信号灯控制的交叉口
* 当前 frame 到停止线距离 < 35 m
* 未来 4 秒内 ego 有可能到达停止线附近
* 地图能提取停止线、斑马线、lane centerline
* 至少有一个相关交互对象：横向车 / 后车 / 行人

### 源类型识别

把 scene 分到以下之一：

* `green_through`
* `yellow_borderline_through`
* `red_wait`
* `red_follow_start`

这一步最好结合 scene 当前 light state 和日志里的 ego 未来趋势一起判断。OpenScene scene/frame 结构支持你直接拿 ego、非 ego 和 map 信息做这些规则。([GitHub][2])

---

# 8. 标签设计

## 8.1 样本级标签

```text
scenario_type
source_type
recovery_type
violation_level
is_complete_event
traffic_policy_type
```

## 8.2 事件级标签

```text
t_red
t_cross_stopline
t_enter_crosswalk
t_enter_conflict_zone
t_recovery_start
t_recovery_end
max_intrusion_distance
```

## 8.3 序列级标签（40 帧）

由于 NAVSIM 原始 OpenScene 是 2 Hz，但官方说明过可以把标注和 ego 轨迹插值到 10 Hz；metric caching 里也这么做。你可以内部用 10 Hz 序列来保存未来 4 秒的事件标签。([GitHub][6])

```text
light_state[k]
zone_state[k]
behavior_state[k]
risk_state[k]
```

其中：

* `k = 1..40`
* `behavior_state`: normal / delayed / violated / correcting_stop / correcting_go / recovered

## 8.4 轨迹监督

```text
traj[40, 3] = (x, y, yaw)
speed[40]
accel[40]
```

这和 NAVSIM 最终需要的 `Trajectory` 接口形式一致。([GitHub][1])

---

# 9. 伪代码

下面这版是按 NAVSIM 的思路写的高层伪代码。

```python
def generate_redlight_counterfactual(scene, frame_idx, map_api, traffic_policy):
    # 1. 读取当前 ego / agents / map
    ego0 = get_ego_state(scene, frame_idx)
    agents0 = get_agents(scene, frame_idx)
    route = get_route_centerline(scene, frame_idx, map_api)
    light_state = get_traffic_light_state(scene, frame_idx)
    stopline = get_next_stopline(route, map_api)
    crosswalk = get_associated_crosswalk(stopline, map_api)
    conflict_zone = get_intersection_conflict_zone(stopline, map_api)

    # 2. 源场景分类
    source_type = classify_source_scene(scene, frame_idx, stopline, light_state)

    # 3. 若不属于候选类型，跳过
    if source_type not in {
        "green_through",
        "yellow_borderline_through",
        "red_wait",
        "red_follow_start",
    }:
        return None

    # 4. 采样违规边界状态
    z = sample_violation_boundary(source_type, ego0, stopline, light_state)

    # 5. 生成阶段 A：到违规边界状态
    traj_A = generate_stage_A_trajectory(
        ego0=ego0,
        route=route,
        stopline=stopline,
        target_boundary=z,
    )

    if not is_kinematically_valid(traj_A):
        return None

    # 6. 根据 z 判定恢复类型
    recovery_type = decide_recovery_type(
        boundary_state=z,
        stopline=stopline,
        crosswalk=crosswalk,
        conflict_zone=conflict_zone,
        rear_agents=select_rear_agents(agents0),
        lateral_agents=select_lateral_agents(agents0),
    )

    # 7. 生成阶段 B：恢复轨迹
    traj_B = generate_stage_B_recovery(
        boundary_state=z,
        recovery_type=recovery_type,
        route=route,
        stopline=stopline,
        crosswalk=crosswalk,
        conflict_zone=conflict_zone,
    )

    full_traj = stitch_and_resample(traj_A, traj_B, horizon_s=4.0, freq_hz=10.0)

    if not is_kinematically_valid(full_traj):
        return None

    # 8. 用 NAVSIM policy 跑 pseudo-sim
    sim_result = run_navsim_pseudosim(
        scene=scene,
        frame_idx=frame_idx,
        ego_trajectory=full_traj,
        traffic_policy=traffic_policy,   # vehicles=IDM, others=log replay
    )

    # 9. 规则过滤
    if not passes_rule_filter(sim_result, recovery_type):
        return None

    # 10. NAVSIM metrics 过滤
    if not passes_metric_filter(sim_result):
        return None

    # 11. 生成标签
    labels = build_labels(
        scene=scene,
        frame_idx=frame_idx,
        full_traj=full_traj,
        sim_result=sim_result,
        source_type=source_type,
        recovery_type=recovery_type,
        boundary_state=z,
    )

    return {
        "ego_trajectory": full_traj,
        "labels": labels,
        "sim_result": sim_result.summary(),
    }
```

---

# 10. 推荐参数表

## 10.1 源场景采样比例

第一版建议：

| source_type               |  比例 |
| ------------------------- | --: |
| green_through             | 35% |
| yellow_borderline_through | 30% |
| red_wait                  | 25% |
| red_follow_start          | 10% |

这个比例是工程建议。原因是：

* `green_through` 和 `yellow_borderline_through` 最容易产生“接近路口误闯”
* `red_wait` 适合做误启动
* `red_follow_start` 更稀有一些，先少量做

## 10.2 边界状态采样

| 参数                       | 推荐范围          |
| ------------------------ | ------------- |
| `t_boundary`             | 0.8 – 2.0 s   |
| `s_rel_stopline`         | -1.0 – +6.0 m |
| `v_boundary`             | 0.5 – 8.0 m/s |
| `red_age`                | 0.1 – 1.0 s   |
| `target_intrusion_depth` | 0.5 – 6.0 m   |

## 10.3 恢复模板比例

| recovery template |  比例 |
| ----------------- | --: |
| stopline_stop     | 40% |
| crosswalk_stop    | 30% |
| commit_go         | 20% |
| false_start_stop  | 10% |

## 10.4 动力学限制

| 参数                  |      默认值 |
| ------------------- | -------: |
| `max_decel_comfort` | 3.0 m/s² |
| `max_decel_hard`    | 5.0 m/s² |
| `max_jerk`          | 4.0 m/s³ |
| `max_lat_offset`    |    0.4 m |
| `max_heading_error` |    8 deg |

这些数值不是 NAVSIM 官方阈值，而是适合你任务的默认起点。NAVSIM v2 的 HC/EC 会对动态不连续和舒适性差的轨迹更敏感，所以限制 jerk 和跨帧动态跳变是必要的。([GitHub][4])

## 10.5 traffic policy

| agent type       | policy       |
| ---------------- | ------------ |
| vehicle          | IDM reactive |
| pedestrian       | log replay   |
| bicycle / others | log replay   |
| static objects   | unchanged    |

---

# 11. Metric 过滤建议

NAVSIM v2 的 EPDMS 强调：

* `TTC`
* `Lane Keeping`
* `History Comfort`
* `Extended Comfort`
* 乘法惩罚项如 driving direction / traffic light compliance 等。([GitHub][4])

所以你的过滤建议分两层：

## 规则层

* 必须发生目标违规
* 必须出现明确恢复
* 不可停在冲突区中央
* 不允许倒车
* 不允许明显离开 route lane

## 指标层

* 无 at-fault collision
* TTC 不低于阈值
* lane keeping 不严重失效
* HC / EC 不崩
* TLC 不能表现成“持续严重无视红灯”，而应体现“误闯后止损/通过修正”

这里最后一条是任务特定解释：因为你本来就是在生成红灯违规样本，所以不能把所有 TLC 不完美样本一票否决；你应该保留“短时违规 + 后续恢复”的样本，过滤掉“长时间持续违规”的样本。这是任务设计上的补充，不是官方 metric 直接支持的语义。

---

# 12. 最小可行版本

如果你现在就要动手，我建议先做这个 MVP：

### 自车

只做两类：

* `green/yellow -> stopline_stop`
* `red_wait -> false_start_stop`

### 他车

* vehicle: IDM reactive
* others: log replay

### 过滤

* 无碰撞
* TTC > 阈值
* HC / EC 基本正常
* 最终停点不进入冲突区深处

### 标签

先只存：

* `source_type`
* `recovery_type`
* `t_cross_stopline`
* `max_intrusion_distance`
* `traj[40,3]`

这样最快能跑起来。

---

# 13. 一句话总结

在 NAVSIM 上，你最合适的做法不是“把灯改红然后强行刹车”，而是：

> **从受控交叉口场景中筛选候选 frame，采样一个未来的红灯违规边界状态 `z_violate`，离线生成一条完整的 4 秒 ego 轨迹，使其先到达该边界状态，再根据可停性选择 stop-recovery 或 go-recovery；同时让 surrounding vehicles 采用 reactive IDM、非车辆保持 log replay，最后用 NAVSIM 的 EPDMS 相关指标和任务专用规则做过滤。** ([GitHub][1])

如果你愿意，我下一步可以直接继续给你一版 **更像代码结构的模块拆分**，例如：
`scene_filter.py / boundary_sampler.py / ego_generator.py / traffic_policy.yaml / label_builder.py`。

[1]: https://github.com/autonomousvision/navsim/blob/main/docs/agents.md?utm_source=chatgpt.com "navsim/docs/agents.md at main · autonomousvision/navsim · GitHub"
[2]: https://github.com/autonomousvision/navsim/blob/main/docs/cache.md?utm_source=chatgpt.com "navsim/docs/cache.md at main · autonomousvision/navsim"
[3]: https://github.com/autonomousvision/navsim/issues/70?utm_source=chatgpt.com "Question about Metrics of NAVSIMv2.0 · Issue #70 - GitHub"
[4]: https://github.com/autonomousvision/navsim/blob/main/docs/metrics.md?utm_source=chatgpt.com "navsim/docs/metrics.md at main - GitHub"
[5]: https://github.com/Sunstroperao/FlowDriveFormer/blob/main/docs/traffic_agents.md?utm_source=chatgpt.com "FlowDriveFormer/docs/traffic_agents.md at main - GitHub"
[6]: https://github.com/autonomousvision/navsim/issues/68?utm_source=chatgpt.com "Is there any Convenient way to extend the navsim dataset from ... - GitHub"
