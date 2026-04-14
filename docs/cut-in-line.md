可以，我给你一个 **专门面向 NAVSIM 的“压线→回正”场景生成方案**。我先做一个明确假设：

**目标不是直接让 ego 撞车，而是生成一种“第一阶段短暂压线，第二阶段主动回正”的高质量边界场景。**
在 NAVSIM 里最实用的做法，是把它设计成一个 **4 秒轨迹模板 / 场景搜索问题**：因为 agent 需要输出 4 秒、10Hz 的未来轨迹；而 v2 评测又会把第一阶段结果和后续 follow-up 场景聚合，所以“先压线、后恢复”这种两阶段设计是契合 NAVSIM 形式的。

---

## 1. 先定清楚你要的“压线”是哪一种

在 NAVSIM 里，我建议你把“压线”定义成：

* **车身短时间跨过车道边界线**
* 但 **不要出 drivable area**
* 也 **不要逆行、闯红灯、碰撞**

原因很现实：
NAVSIM 的 EPDMS 里，`NC / DAC / DDC / TLC` 这类项更像“硬门槛”或强过滤；而 `LK / TTC / comfort / progress` 才更适合表达“有挑战但不彻底失真”的边界行为。所以你要做的是一个 **轻度违规 / 边界失稳** 场景，而不是把场景直接做坏。

所以目标应当是：

**压线惩罚主要体现在 lane keeping 上，而不是 drivable-area、direction 或 collision 上。**

---

## 2. 最推荐的总框架：双层生成

你可以把这个问题写成一个双层问题：

### 外层：选场景、选压线模板参数

从 NAVSIM 数据集里挑那些“适合发生压线但可恢复”的 scene，然后给每个 scene 采样一组参数：

[
\theta={side, T_1, T_2, \delta, v_{scale}, a_{recover}}
]

含义是：

* `side`：向左压还是向右压
* `T1`：第一阶段结束时刻，通常 1.2 到 2.0 秒
* `T2`：第二阶段恢复完成时刻，通常 3.0 到 4.0 秒
* `δ`：压线深度
* `v_scale`：纵向速度缩放
* `a_recover`：恢复阶段的纵向减速 / 稳定化强度

### 内层：给定场景和参数，构造两阶段 ego 目标轨迹

在 route 参考线的 Frenet 坐标里，生成：

* 第一阶段：横向偏移增大，发生压线
* 第二阶段：横向偏移回归 0，航向回正

然后再把这条轨迹作为：

* 你的 planner 的直接输出模板，或者
* 你的 scenario search 的“目标行为约束”

---

## 3. 轨迹模板怎么定义最稳

### 用 Frenet 表示

沿参考路线中心线建立 Frenet 坐标：

* `s(t)`：沿路线前进距离
* `d(t)`：相对路线中心线的横向偏移

这样“压线”和“回正”都很好写。

---

### 第一阶段：压线

设当前在车道中心附近，`d(0) ≈ 0`。
车道边界在 `±w_lane/2`，车辆半宽是 `w_car/2`。

如果你想要“车身压线”，不需要 ego 中心跑到边界外很多，只需要：

[
|d_{peak}| > \frac{w_{lane}}{2} - \frac{w_{car}}{2} + \epsilon
]

这里 `ε` 可以取 0.05 到 0.20 米，表示“明确压到线，不只是擦边”。

所以第一阶段目标峰值可以设成：

[
d_{peak} = sign(side)\cdot\left(\frac{w_{lane}}{2}-\frac{w_{car}}{2}+\epsilon\right)
]

然后在 `t ∈ [0, T1]` 上，用 quintic polynomial 从 `d=0` 平滑过渡到 `d=d_peak`，并令：

* 初始横向速度、加速度为 0
* 到 `T1` 时横向速度接近 0 或略小于 0

这样轨迹不会很抖。

---

### 第二阶段：回正

在 `t ∈ [T1, T2]` 上，再用一个 quintic：

[
d(T_1)=d_{peak}\rightarrow d(T_2)=0
]

并约束：

* `\dot d(T2)=0`
* `\ddot d(T2)=0`
* 末端 heading 和 route tangent 对齐

这就是“回正”。

---

### 纵向速度怎么配

纵向不要太激进。建议：

* 第一阶段：维持当前速度或轻微减速
* 第二阶段：略微减速，帮助横摆稳定和回正

一个简单做法是让 `s(t)` 跟 baseline 速度曲线走，只在第二阶段乘一个 `0.9 ~ 0.97` 的缩放。
这样 recovery 会比“继续高速前冲”稳定很多。

---

## 4. 在 NAVSIM 里怎样挑适合生成这种场景的原始 scene

这一步特别重要。不是所有 scene 都适合“压线再回正”。

建议只从这类场景里挖：

### 适合的 scene

* ego 前方 4 秒内路线清晰、无遮挡
* 左右至少一侧有相邻 lane 或足够 drivable shoulder
* 压线侧不是 curb / barrier 紧贴
* 没有很近的静态障碍物
* 不是红灯停车线前 10 到 20 米
* 不是高密度交叉口冲突流
* 当前 ego 速度中等，通常 5 到 15 m/s 更容易做出可恢复压线

### 不适合的 scene

* 紧贴路缘石
* 路口中央复杂冲突区
* 前方急弯且 lane width 窄
* 紧邻行人、自行车或停靠车辆
* 高速大曲率变道区

原因很简单：
你要的是 **“可恢复的边界行为”**，而不是一压线就直接触发 DAC / NC / DDC 崩掉。EPDMS 会对这类严重失真很不友好。

---

## 5. 评分函数怎么写：让它“压得出来，也回得回来”

我建议你把生成器的打分函数写成：

[
S(\theta)=
w_1 S_{encroach}
+w_2 S_{recover}
-w_3 P_{hardfail}
-w_4 P_{comfort}
-w_5 P_{oscillation}
]

### (1) 压线得分 `S_encroach`

衡量第一阶段压线是否成功：

* 压线开始时间是否在目标区间内
* 压线持续时间是否达标
* 压线深度是否达到目标
* 压线主要发生在 `0~T1`

可定义为：

[
S_{encroach} =
\alpha_1 \cdot duration_{line_overlap}
+\alpha_2 \cdot max(0, |d_{peak}|-d_{thr})
]

---

### (2) 回正得分 `S_recover`

衡量第二阶段恢复质量：

* `|d(T2)|` 足够小
* `|heading\_error(T2)|` 足够小
* `|\dot d(T2)|` 足够小
* 恢复阶段没有二次摆动

例如：

[
S_{recover}=
-\beta_1 |d(T_2)|
-\beta_2 |\psi(T_2)-\psi_{ref}(T_2)|
-\beta_3 |\dot d(T_2)|
]

---

### (3) 硬失败惩罚 `P_hardfail`

这个必须重罚：

* 碰撞
* 出 drivable area
* 明显逆向
* 闯红灯

因为这些在 NAVSIM 指标里本来就是很致命的项。

---

### (4) 舒适性惩罚 `P_comfort`

* 横向加速度
* jerk
* 曲率变化率

你不想得到一个“技术上压线了，但人类绝不会这么开”的轨迹。

---

### (5) 振荡惩罚 `P_oscillation`

这个对“回正”尤其重要。
要防止第二阶段变成：

* 先回正
* 又反向摆一下
* 再回来

可以直接惩罚 `d(t)` 在 `T1~T2` 内的符号变化次数或局部极值数量。

---

## 6. 两种具体落地方式

---

### 方案 A：做成“ego 目标轨迹生成器”

这是最简单、最快能跑通的。

#### 做法

你直接写一个 `TwoPhaseLaneEncroachmentAgent(AbstractAgent)`，在 `compute_trajectory()` 里：

1. 读 route centerline
2. 根据 scene 几何判断可压线方向
3. 采样若干组 `θ`
4. 生成两阶段 Frenet proposal
5. 用上面的 score 排序
6. 返回分最高的 proposal

#### 适用场景

* 你先验证“这种两阶段模板在 NAVSIM 里能否稳定存在”
* 你想先做一个可控 benchmark
* 你想研究 planner 的恢复能力

这完全符合 NAVSIM agent 接口，因为 agent 的核心就是实现 `compute_trajectory()` 并输出 4 秒轨迹。

---

### 方案 B：做成“场景搜索器”

这是更接近你前面说的 scenario generation。

#### 做法

不是直接规定 ego 去压线，而是：

1. 先从数据集中挖 scene
2. 对背景 actor 做轻微扰动

   * 前车轻微减速
   * 侧车轻微侵占
   * 邻车道 blocker
3. 重新跑你的 ego planner
4. 检查 ego 是否出现“第一阶段压线、第二阶段回正”
5. 满足模板就保留这个 scene

#### 优点

* 更像真实场景生成
* 更适合做 adversarial / robustness evaluation

#### 缺点

* 工程量大
* NAVSIM 官方主接口本身更偏重“给定输入评 ego 轨迹”，所以你通常要自己做 scene preprocessor 或扩展 evaluator，而不是只写 agent。

---

## 7. 我最推荐的时间分配

在 NAVSIM 一个 horizon 是 4 秒，所以你最实用的切法是：

* **阶段 1：0.0 到 1.6 秒**
  从中心线逐渐偏到压线
* **阶段 2：1.6 到 4.0 秒**
  平滑回正并稳定

为什么不是 2 秒 + 2 秒死切？

因为真实恢复通常比偏离更慢。
如果第一阶段拖太久，第二阶段容易来不及完全回正，末端 heading 和 lateral error 会不好看。

你可以从这组初值开始：

* `T1 = 1.4, 1.6, 1.8`
* `T2 = 3.2, 3.6, 4.0`
* `ε = 0.05, 0.10, 0.15`
* `v_scale = 1.0, 0.95, 0.90`

---

## 8. 一个非常关键的细节：不要把“压线”做成“出界”

这是 NAVSIM 里最值得提前规避的坑。

你想要的是：

* **压到 lane boundary**
* 但最好 **仍在 drivable polygon 内**

因为如果直接把 ego 做出可行驶区域，DAC 那类项会很伤，最后这个样本就不再是“边界场景”，而是“明显坏轨迹”。NAVSIM v2 的指标就是为了区分这种情况。

所以一个很好的生成准则是：

**允许 lane keeping 变差，不允许 drivable-area 崩掉。**

---

## 9. 结合 NAVSIM v2，两阶段怎么和评测配合

NAVSIM v2 使用 two-stage pseudo closed-loop aggregation，所以你设计的“第一阶段压线、第二阶段恢复”除了能写进单条 4 秒轨迹外，还可以进一步加一个要求：

**阶段 1 末端状态必须是“可恢复状态”**。

也就是在 `t=T1` 时：

* lateral offset 已经达到峰值
* 但 yaw 还没有完全发散
* 速度还在可控范围
* 周围 TTC 没被压到极低

这样 follow-up 场景里 recovery 才更自然。这个思路和 v2 的 staged evaluation 是一致的。

---

## 10. 你可以直接照着写的模块划分

### 模块 1：SceneMiner

输入 NAVSIM scene，输出候选 scene 列表
过滤条件：

* 单侧可压线
* 无 curb 紧逼
* 中低密度
* 4 秒 route 可追踪

### 模块 2：BoundaryTemplate

给定 route 和 `θ`，生成 Frenet 两阶段目标：

* stage1 encroach
* stage2 recover

### 模块 3：ProposalRollout

把 Frenet 转成 Cartesian `(x, y, yaw)`
并检查：

* 动力学可行性
* drivable area
* collision
* traffic light

### 模块 4：ScenarioScorer

计算：

* `encroach score`
* `recover score`
* `hard fail penalties`
* `comfort penalties`

### 模块 5：DatasetBuilder

保留满足下面条件的样本：

* 成功压线
* 成功回正
* 无碰撞
* 不出 drivable area
* 可恢复性好

---

## 11. 我建议的样本验收标准

你最后保留的数据，建议至少满足：

[
duration_{encroach} \ge 0.3s
]

[
|d(T_2)| < 0.15m
]

[
|\psi(T_2)-\psi_{ref}(T_2)| < 3^\circ
]

并且：

* 无 NC
* 无 DAC
* 无 DDC
* 无 TLC

这样得到的样本会更像“真实边界失稳然后纠正”，而不是单纯造坏。NAVSIM 指标体系本身也更偏好这种区别。

---

## 12. 我最推荐你的第一版

别一开始就做背景车扰动。
**第一版直接做“ego 两阶段轨迹生成器”**，因为最容易验证你的模板是否成立：

1. 先在 NAVSIM agent 接口里生成 4 秒两阶段轨迹。
2. 跑本地 evaluation，看 LK 会不会下降但 DAC/DDC/NC 仍保持健康。
3. 跑通后，再把它升级成“通过背景 actor 扰动诱导 ego 出现两阶段行为”的 scenario generator。

这样路线最稳。

如果你愿意，我下一条直接给你写一版 **NAVSIM 中“压线→回正”两阶段轨迹生成器的伪代码骨架**，包括 `compute_trajectory()`、Frenet 模板和 score 函数。
