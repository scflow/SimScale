可以。下面给你一套 **在 NAVSIM v2 下可落地的实现方案**，目标是：

* **ego** 用 MOBIL 做横向换道决策
* **背景车** 也用同一套 MOBIL 逻辑做 reactive traffic
* 最终严格接到 NAVSIM 现有扩展口：

  * ego：`AbstractAgent.compute_trajectory()`
  * 背景车：`traffic_agents_policy`
* 输出符合 NAVSIM 的 `Trajectory` / 仿真 rollout 方式。NAVSIM 官方 agent 文档要求 agent 重写 `compute_trajectory()` 并返回 `navsim.common.dataclasses.Trajectory`；PDM 评分默认评 4 秒、10Hz。NAVSIM v2.0 已支持 reactive traffic agent policies，v2.1 又加入了 two-stage reactive traffic agents。([GitHub][1])

---

## 1. 先说结论：最合适的工程结构

建议你这样放：

```text
navsim/
  behavior/
    idm.py
    mobil.py
    lane_change_trajectory.py
    scene_adapter.py
  agents/
    mobil_agent.py
  traffic_agents_policies/
    mobil_traffic_agents.py
  planning/script/config/common/agent/
    mobil_agent.yaml
  planning/script/config/common/traffic_agents_policy/
    mobil_traffic_agents.yaml
```

这样做的原因是：

* `mobil.py` 只做“换不换、往哪边换”
* `idm.py` 只做纵向加速度
* `lane_change_trajectory.py` 负责把离散动作变成连续轨迹
* `mobil_agent.py` 只是 ego 适配层
* `mobil_traffic_agents.py` 只是背景车适配层

NAVSIM 仓库本身就把 ego agent 放在 `navsim/agents/` 下，traffic policy 放在 `navsim/traffic_agents_policies/` 下；评估脚本配置也通过 Hydra 的 `agent:` 默认项加载 agent，命令行里也支持传 `traffic_agents_policy=...`。([GitHub][2])

---

## 2. 你要实现的“数学核心”

### 2.1 IDM：纵向控制

MOBIL 本身不是完整驾驶模型，它通常建立在某个跟驰模型之上，最常见就是 IDM。标准 IDM 加速度写法是：

[
a_{\text{idm}} = a \left[1-\left(\frac{v}{v_0}\right)^\delta-\left(\frac{s^*(v,\Delta v)}{s}\right)^2\right]
]

其中期望间距：

[
s^*(v,\Delta v)=s_0 + vT + \frac{v\Delta v}{2\sqrt{ab}}
]

这里：

* (v)：自车速度
* (v_0)：期望速度
* (s)：与前车间距
* (\Delta v = v - v_{\text{lead}})：接近速度
* (s_0)：静止最小间距
* (T)：期望车头时距
* (a)：最大舒适加速度
* (b)：舒适减速度
* (\delta)：通常取 4。([维基百科][3])

### 2.2 MOBIL：横向换道判决

MOBIL 的思想是：**换道后既要安全，也要值得。**

#### 安全约束

目标车道后车在你换过去之后，不能被迫急刹得太狠：

[
a'*{\text{rear,new}} > -b*{\text{safe}}
]

其中 (a'_{\text{rear,new}}) 是目标车道后车在你完成换道后的纵向加速度。这个安全项是 MOBIL 的核心约束之一。([马丁·特雷伯研究所][4])

#### 激励约束

换道收益要超过阈值：

[
\Delta a_{\text{ego}} + p\left(\Delta a_{\text{rear,new}} + \Delta a_{\text{rear,old}}\right) > \Delta a_{\text{th}}
]

常见写法也可展开为：

[
(a'*{\text{ego}}-a*{\text{ego}})

* p\Big[(a'*{\text{rear,new}}-a*{\text{rear,new}})
* (a'*{\text{rear,old}}-a*{\text{rear,old}})\Big]

> \Delta a_{\text{th}}
> ]

其中：

* (p)：politeness factor，礼让系数
* (\Delta a_{\text{th}})：换道阈值
* (a'_{\text{ego}})：换道后 ego 加速度
* (a_{\text{ego}})：当前车道 ego 加速度。([马丁·特雷伯研究所][4])

### 2.3 建议参数

第一版先用这组：

```python
IDM:
v0 = 13.9      # 50 km/h
T = 1.5
s0 = 2.0
a = 1.5
b = 2.0
delta = 4

MOBIL:
p = 0.3
a_thr = 0.2
b_safe = 4.0
cooldown_s = 3.0
route_bias = 0.2
```

这组参数偏保守，适合先跑通，再调 aggressive / conservative 风格。

---

## 3. 决策链路怎么跑

每个决策周期，对某辆车都走同一套流程：

### Step A：提取 lane-neighbor 关系

对当前车，构造 6 个邻居槽位：

* 当前车道前车 `curr_front`
* 当前车道后车 `curr_rear`
* 左车道前车 `left_front`
* 左车道后车 `left_rear`
* 右车道前车 `right_front`
* 右车道后车 `right_rear`

### Step B：分别评估 keep / left / right

* `keep`：当前车道用 IDM 算当前纵向加速度
* `left`：检查左车道存在、可达、安全约束、激励约束
* `right`：同理

### Step C：route-aware 修正

NAVSIM 会给 ego 一个离散 driving command：left / straight / right / unknown，而且文档明确说 left/right 覆盖转弯、换道和急弯。这个高层命令非常适合拿来给 ego 的 MOBIL 加 route bias。([GitHub][1])

例如：

* command = left：左换道激励加 `+route_bias`
* command = right：右换道激励加 `+route_bias`
* command = straight：左右两边都不加偏置，甚至可加轻微惩罚

### Step D：输出动作

输出：

```python
KEEP
CHANGE_LEFT
CHANGE_RIGHT
```

再附带：

```python
a_lon      # 纵向加速度
target_lane_id
```

### Step E：轨迹生成

把动作变成连续轨迹，而不是只输出动作标签。因为 NAVSIM 的 agent 最终必须返回未来 BEV poses 组成的 `Trajectory`。官方文档对这一点写得很清楚。([GitHub][1])

---

## 4. 具体实现位置

### 4.1 `navsim/behavior/idm.py`

```python
from dataclasses import dataclass
import math

@dataclass
class IDMParams:
    v0: float = 13.9
    T: float = 1.5
    s0: float = 2.0
    a: float = 1.5
    b: float = 2.0
    delta: int = 4

class IDM:
    def __init__(self, params: IDMParams):
        self.p = params

    def accel(self, v: float, gap: float, dv: float) -> float:
        # dv = v - v_lead
        gap = max(gap, 0.1)
        s_star = self.p.s0 + v * self.p.T + v * dv / (2.0 * math.sqrt(self.p.a * self.p.b))
        return self.p.a * (1.0 - (v / self.p.v0) ** self.p.delta - (s_star / gap) ** 2)

    def free_accel(self, v: float) -> float:
        return self.p.a * (1.0 - (v / self.p.v0) ** self.p.delta)
```

---

### 4.2 `navsim/behavior/mobil.py`

```python
from dataclasses import dataclass

@dataclass
class MobilParams:
    politeness: float = 0.3
    accel_threshold: float = 0.2
    safe_decel: float = 4.0
    cooldown_s: float = 3.0
    route_bias: float = 0.2

class MobilModel:
    def __init__(self, params: MobilParams, idm):
        self.p = params
        self.idm = idm

    def lane_change_gain(
        self,
        ego_now, ego_after,
        new_rear_now, new_rear_after,
        old_rear_now, old_rear_after,
        route_bonus: float = 0.0,
    ) -> float:
        return (
            (ego_after - ego_now)
            + self.p.politeness * (
                (new_rear_after - new_rear_now)
                + (old_rear_after - old_rear_now)
            )
            + route_bonus
        )

    def safe(self, new_rear_after: float) -> bool:
        return new_rear_after > -self.p.safe_decel

    def decide(self, candidate_scores: dict) -> str:
        # candidate_scores = {"KEEP": 0.0, "LEFT": gain_left, "RIGHT": gain_right}
        best_action = max(candidate_scores, key=candidate_scores.get)
        if candidate_scores[best_action] <= self.p.accel_threshold:
            return "KEEP"
        return best_action
```

---

### 4.3 `navsim/behavior/scene_adapter.py`

这个文件负责把 NAVSIM 的 scene / agent_input / traffic objects，统一转成你自己的轻量数据结构，避免 MOBIL 直接绑定 NAVSIM 内部类。

```python
from dataclasses import dataclass

@dataclass
class ActorState:
    x: float
    y: float
    yaw: float
    v: float
    a: float
    lane_id: str | None
    length: float = 4.8
    width: float = 2.0

@dataclass
class NeighborSet:
    curr_front: ActorState | None = None
    curr_rear: ActorState | None = None
    left_front: ActorState | None = None
    left_rear: ActorState | None = None
    right_front: ActorState | None = None
    right_rear: ActorState | None = None
```

这个适配层是你后面最省事的部分。ego 和背景车都先转成这个抽象，再喂给 IDM/MOBIL。

---

### 4.4 `navsim/behavior/lane_change_trajectory.py`

这里做两件事：

1. 纵向：用 `a_lon` rollout 速度和弧长
2. 横向：把 `KEEP / LEFT / RIGHT` 映射到目标 lane centerline，再平滑过渡

推荐简化公式：

#### 横向偏移

用五次多项式或者 sigmoid 做车道中心切换。第一版可以直接用 smoothstep：

[
r(\tau)=10\tau^3 - 15\tau^4 + 6\tau^5,\quad \tau\in[0,1]
]

然后：

[
d(t)=d_0 + r(t/t_{lc}) (d_{\text{target}}-d_0)
]

其中 (t_{lc}) 是换道持续时间，建议 2.5 到 3.5 秒。

#### 纵向 rollout

[
v_{k+1} = \max(0, v_k + a_{\text{lon}} \Delta t)
]
[
s_{k+1} = s_k + v_k \Delta t + \frac{1}{2} a_{\text{lon}} \Delta t^2
]

再把 ((s, d)) 投回目标车道中心线，生成 ((x, y, \psi))。

---

## 5. ego 侧怎么接 NAVSIM

### 5.1 文件位置

放在：

```text
navsim/agents/mobil_agent.py
```

### 5.2 为什么这么接

NAVSIM 文档规定 agent 需要继承 `AbstractAgent`，并实现：

* `name()`
* `initialize()`
* `get_sensor_config()`
* `compute_trajectory()`

`compute_trajectory()` 要返回 `Trajectory`；`ConstantVelocityAgent` 是最简单模板。([GitHub][1])

### 5.3 代码骨架

```python
import numpy as np
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory

from navsim.behavior.idm import IDM, IDMParams
from navsim.behavior.mobil import MobilModel, MobilParams
from navsim.behavior.scene_adapter import build_ego_state_and_neighbors
from navsim.behavior.lane_change_trajectory import rollout_ego_trajectory

class MobilAgent(AbstractAgent):
    requires_scene = False

    def __init__(self,
                 trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.1)):
        super().__init__(trajectory_sampling)
        self.idm = IDM(IDMParams())
        self.mobil = MobilModel(MobilParams(), self.idm)

    def name(self) -> str:
        return self.__class__.__name__

    def initialize(self) -> None:
        pass

    def get_sensor_config(self) -> SensorConfig:
        return SensorConfig.build_no_sensors()

    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        ego, neighbors, route_cmd, lane_graph = build_ego_state_and_neighbors(agent_input)

        action, a_lon, target_lane = decide_with_mobil(
            ego=ego,
            neighbors=neighbors,
            route_cmd=route_cmd,
            lane_graph=lane_graph,
            idm=self.idm,
            mobil=self.mobil,
        )

        poses = rollout_ego_trajectory(
            ego=ego,
            action=action,
            a_lon=a_lon,
            target_lane=target_lane,
            sampling=self._trajectory_sampling,
            lane_graph=lane_graph,
        )

        return Trajectory(np.asarray(poses, dtype=np.float32), self._trajectory_sampling)
```

### 5.4 ego 版本的现实边界

这个方案 **本地评估完全可行**。但如果你未来想上 leaderboard，NAVSIM 文档明确写了测试时只会给传感器历史，不会给 maps / tracks / occupancy 这类训练时可用的 privileged 信息。也就是说，使用精确 lane/topology/neighbor tracks 的 MOBIL 更适合作为本地研究版或 privileged baseline。([GitHub][1])

---

## 6. 背景车怎么接 NAVSIM

### 6.1 文件位置

放在：

```text
navsim/traffic_agents_policies/mobil_traffic_agents.py
```

### 6.2 为什么这么接

NAVSIM v2.0 release 明确说加入了 reactive traffic agent policies，v2.1 进一步加入 two-stage reactive traffic agents。社区 issue 里也能看到 `run_pdm_score.py` 命令直接支持 `traffic_agents_policy=...`，并提到了内置 policy 名 `navsim_IDM_traffic_agents` 和 `log_replay_traffic_agents`。([GitHub][5])

### 6.3 具体做法

你的 `mobil_traffic_agents.py` 最好去模仿内置的 `navsim_IDM_traffic_agents` 思路：

* 每个 simulation step 遍历可控背景车
* 为每辆车构造 `ActorState + NeighborSet`
* 用同一套 `decide_with_mobil()`
* 用同一套 rollout 更新未来一步或短时状态

代码逻辑上像这样：

```python
class MobilTrafficAgentsPolicy(...):
    def __init__(self, ...):
        self.idm = IDM(IDMParams())
        self.mobil = MobilModel(MobilParams(), self.idm)

    def step(self, traffic_state, ego_trajectory, map_api, dt):
        updated_agents = []

        for agent in traffic_state.vehicles:
            actor, neighbors, lane_graph = build_background_state_and_neighbors(
                agent, traffic_state, map_api
            )

            action, a_lon, target_lane = decide_with_mobil(
                ego=actor,
                neighbors=neighbors,
                route_cmd=None,
                lane_graph=lane_graph,
                idm=self.idm,
                mobil=self.mobil,
            )

            next_state = rollout_background_agent_one_step(
                actor=actor,
                action=action,
                a_lon=a_lon,
                target_lane=target_lane,
                dt=dt,
                lane_graph=lane_graph,
            )
            updated_agents.append(next_state)

        return updated_agents
```

---

## 7. 共享决策函数应该怎么写

这是最核心的公共入口：

```python
def decide_with_mobil(ego, neighbors, route_cmd, lane_graph, idm, mobil):
    # 当前车道纵向加速度
    curr_gap = compute_gap(ego, neighbors.curr_front)
    curr_dv = ego.v - neighbors.curr_front.v if neighbors.curr_front else 0.0
    a_keep = idm.accel(ego.v, curr_gap, curr_dv)

    scores = {"KEEP": 0.0}
    outputs = {"KEEP": (a_keep, ego.lane_id)}

    for action, front, rear, lane_id in [
        ("LEFT", neighbors.left_front, neighbors.left_rear, left_lane_id(ego, lane_graph)),
        ("RIGHT", neighbors.right_front, neighbors.right_rear, right_lane_id(ego, lane_graph)),
    ]:
        if lane_id is None:
            scores[action] = -1e9
            continue

        # ego 换道前后
        ego_now = a_keep
        gap_new = compute_gap(ego, front)
        dv_new = ego.v - front.v if front else 0.0
        ego_after = idm.accel(ego.v, gap_new, dv_new)

        # 新后车 before / after
        new_rear_now = rear_free_or_following_accel(rear, front, idm)
        new_rear_after = rear_free_or_following_accel(rear, ego, idm)

        # 原后车 before / after
        old_rear_now = rear_free_or_following_accel(neighbors.curr_rear, ego, idm)
        old_rear_after = rear_free_or_following_accel(neighbors.curr_rear, neighbors.curr_front, idm)

        if not mobil.safe(new_rear_after):
            scores[action] = -1e9
            continue

        route_bonus = compute_route_bonus(route_cmd, action, mobil.p.route_bias)
        gain = mobil.lane_change_gain(
            ego_now, ego_after,
            new_rear_now, new_rear_after,
            old_rear_now, old_rear_after,
            route_bonus,
        )

        scores[action] = gain
        outputs[action] = (ego_after, lane_id)

    best = mobil.decide(scores)
    a_lon, target_lane = outputs[best]
    return best, a_lon, target_lane
```

---

## 8. YAML 配置怎么加

### 8.1 ego agent 配置

放在：

```text
navsim/planning/script/config/common/agent/mobil_agent.yaml
```

示例：

```yaml
_target_: navsim.agents.mobil_agent.MobilAgent
_convert_: all

trajectory_sampling:
  _target_: nuplan.planning.simulation.trajectory.trajectory_sampling.TrajectorySampling
  time_horizon: 4.0
  interval_length: 0.1
```

之所以要放在这里，是因为 `default_run_pdm_score.yaml` 的 `defaults` 里已经有 `- agent: constant_velocity_agent` 这个 Hydra 入口。你新增一个同目录 yaml 后，就可以通过 `agent=mobil_agent` 切换。([GitHub][6])

### 8.2 traffic policy 配置

放在：

```text
navsim/planning/script/config/common/traffic_agents_policy/mobil_traffic_agents.yaml
```

示例：

```yaml
_target_: navsim.traffic_agents_policies.mobil_traffic_agents.MobilTrafficAgentsPolicy
_convert_: all

dt: 0.1
two_stage: true
```

---

## 9. 运行方式

最小测试命令：

```bash
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
  train_test_split=navtest \
  experiment_name=mobil_debug \
  agent=mobil_agent \
  traffic_agents_policy=mobil_traffic_agents \
  metric_cache_path=$NAVSIM_EXP_ROOT/metric_cache
```

NAVSIM 的默认 PDM 配置文件里使用 Hydra，默认 agent 是 `constant_velocity_agent`；issue 里也展示了 `run_pdm_score.py ... traffic_agents_policy=$POLICY` 的实际调用方式。([GitHub][6])

---

## 10. 第一版强烈建议的简化

别一上来就做全功能版本。第一版建议这样：

### ego

* 用 `get_sensor_config() -> build_no_sensors()`
* 从 ego status + route command + privileged lane/tracks 构造邻居
* MOBIL + IDM + 简化 lane-change rollout
* 输出 `Trajectory`

### 背景车

* 只控制 vehicle 类 agent
* 每步做 one-step rollout
* lane change duration 固定 3 秒
* 加 3 秒 cooldown，防止左右横跳

这样你能最快看见：

* ego progress
* TTC
* comfort
* no-at-fault collision

---

## 11. 最容易踩的坑

### 坑 1：只写 MOBIL，不写 IDM

这样你只能得到 `LEFT/RIGHT/KEEP`，但 ego 需要的是连续未来轨迹，背景车也需要连续 rollout。

### 坑 2：直接横向瞬移到新车道

PDM 对 comfort 很敏感，横向必须平滑，不然 jerk / heading 变化会很难看。

### 坑 3：不加 route bias

NAVSIM 的 driving command 是 route-based，而且 left/right 明确覆盖 lane changes。ego 如果无视这个高层意图，分数通常会掉。([GitHub][1])

### 坑 4：ego 研究版和 leaderboard 版混在一起

本地研究版可以吃 privileged tracks/map；提交版要受测试输入限制。NAVSIM 文档已经明确说明测试时不会给 maps/tracks/occupancy。([GitHub][1])

---

## 12. 最终推荐方案

一句话总结：

**在 NAVSIM 下，实现“ego + 背景车都用 MOBIL”的最优方案是：**

* 用 `navsim/behavior/idm.py` 实现纵向 IDM
* 用 `navsim/behavior/mobil.py` 实现横向 MOBIL
* 用 `navsim/behavior/lane_change_trajectory.py` 把动作转成轨迹
* 用 `navsim/agents/mobil_agent.py` 接 ego 的 `compute_trajectory()`
* 用 `navsim/traffic_agents_policies/mobil_traffic_agents.py` 接 reactive background traffic
* 用 Hydra yaml 把 `agent=mobil_agent` 和 `traffic_agents_policy=mobil_traffic_agents` 挂到 `run_pdm_score.py`

这套方案和 NAVSIM v2 的官方扩展口是对齐的，既符合框架，也方便后面调参和做 ablation。([GitHub][1])

下一条我可以直接给你一版 **可复制的代码骨架**，按这几个文件分别展开。

[1]: https://github.com/autonomousvision/navsim/blob/main/docs/agents.md "https://github.com/autonomousvision/navsim/blob/main/docs/agents.md"
[2]: https://github.com/autonomousvision/navsim/blob/main/navsim/agents/constant_velocity_agent.py "https://github.com/autonomousvision/navsim/blob/main/navsim/agents/constant_velocity_agent.py"
[3]: https://en.wikipedia.org/wiki/Intelligent_driver_model "https://en.wikipedia.org/wiki/Intelligent_driver_model"
[4]: https://mtreiber.de/publications/MOBIL_TRB.pdf "https://mtreiber.de/publications/MOBIL_TRB.pdf"
[5]: https://github.com/autonomousvision/navsim/releases "https://github.com/autonomousvision/navsim/releases"
[6]: https://github.com/autonomousvision/navsim/blob/main/navsim/planning/script/config/pdm_scoring/default_run_pdm_score.yaml "https://github.com/autonomousvision/navsim/blob/main/navsim/planning/script/config/pdm_scoring/default_run_pdm_score.yaml"
