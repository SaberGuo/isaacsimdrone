# OmniPerception IsaacDrone Test6 程序逻辑结构文档

## 一、整体架构概览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        test6_train_skrl.py (入口脚本)                        │
│                    基于 SKRL 框架的 PPO 强化学习训练脚本                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     MyDroneRLEnv (omniperception_isaacdrone)                 │
│                    继承自 IsaacLab 的 ManagerBasedRLEnv                      │
│  文件: source/omniperception_isaacdrone/envs/test6_env.py                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        ▼                             ▼                             ▼
┌───────────────┐            ┌───────────────┐            ┌───────────────┐
│   场景管理     │            │   MDP 管理器   │            │   自定义缓冲区   │
│  (SceneCfg)   │            │ (Managers)    │            │  (Buffers)    │
└───────────────┘            └───────────────┘            └───────────────┘
```

---

## 二、入口脚本层 (scripts/test6_train_skrl.py)

### 2.1 主要组件

| 组件 | 功能描述 |
|------|----------|
| **命令行参数** | 定义训练超参数：state_dim(17), lidar_dim(432), PPO参数, Dijkstra/APF奖励参数 |
| **WallSpawner** | 生成工作空间边界墙 (±80m 范围，6个面) |
| **setup_global_obstacles** | 预生成全局共享障碍物模板到 `/World/Obstacles` |
| **SkrlSpaceAdapter** | 环境包装器，适配 SKRL 的 observation/action 空间 |
| **StructuredFeatureExtractor** | 策略网络特征提取器：state(17D) + lidar(432D) → feat(256D) |
| **Policy/Value** | 高斯策略网络和价值网络 |
| **训练循环** | 标准 PPO 训练流程，支持 TensorBoard 日志、梯度监控、课程学习跟踪 |

### 2.2 神经网络结构

```
StructuredFeatureExtractor:
  State分支: 17 → LayerNorm → Linear(128) → Tanh → Linear(128) → Tanh
  Lidar分支: 432 → LayerNorm → Linear(256) → Tanh → Linear(256) → Tanh
  融合: Concat(128+256=384) → Linear(256) → Tanh

Policy:  feat(256) → Linear(4) → Tanh → mean,  learnable log_std
Value:   feat(256) → Linear(1) → value
```

---

## 三、环境注册 (tasks/test6_registry.py)

```python
# 两个注册的环境
Isaac-OmniPerception-Drone-v0        → Test6DroneEnvCfg      (无激光雷达)
Isaac-OmniPerception-Drone-Lidar-v0  → Test6DroneLidarEnvCfg (带激光雷达)

# 入口点
entry_point = "omniperception_isaacdrone.envs.test6_env:MyDroneRLEnv"
```

---

## 四、环境配置详解 (envs/test6_env_cfg.py)

### 4.1 配置类层次

```
NormalizationCfg                    # 归一化超参数
ObstacleCurriculumSettingsCfg       # 课程学习设置
Test6SceneCfg / Test6SceneWithLidarCfg   # 场景配置
Test6ActionsCfg                     # 动作配置
Test6ObservationsCfg                # 观察配置
Test6EventCfg                       # 事件配置
Test6RewardsCfg                     # 奖励配置（核心）
Test6TerminationsCfg                # 终止条件
Test6CurriculumCfg                  # 课程学习
Test6DroneEnvCfg / Test6DroneLidarEnvCfg # 主配置类
```

### 4.2 场景配置 (Test6SceneCfg)

| 元素 | 配置 |
|------|------|
| terrain | 平面地形，摩擦系数1.0 |
| robot | 无人机机器人，初始位置 (0,0,5) |
| contact_sensor | 碰撞检测传感器 |
| dome_light / distant_light | 光照 |
| obstacles | 全局障碍物，`/World/Obstacles/obj_.*` |
| lidar (可选) | 激光雷达传感器 |

### 4.3 观察空间配置 (17D State + 432D Lidar)

```python
PolicyCfg:
  - root_pos_z (1D):  归一化高度位置
  - root_quat (4D):   归一化四元数 (w,x,y,z)
  - root_lin_vel (3D): 归一化线速度
  - root_ang_vel (3D): 归一化角速度
  - projected_gravity (3D): 投影重力向量
  - goal_delta (3D):  归一化目标方向向量
  - lidar_grid (432D): 激光雷达最小距离网格 (6×72 bins)

Total: 17 + 432 = 449D
```

### 4.4 奖励配置详解 (Test6RewardsCfg)

#### 密集导航奖励（主信号）

| 奖励项 | 权重 | 说明 |
|--------|------|------|
| `progress_to_goal` | 40.0 | 向目标前进的进度奖励 (delta_distance/dt) |
| `dist_to_goal` | 5.0 | 距离目标的指数衰减奖励 exp(-0.5*(d/std)^2) |
| `vel_towards_goal` | 8.0 | 朝向目标的速度投影奖励 |
| `dijkstra_progress` | 20.0 | Dijkstra测地距离进度奖励（核心创新） |

#### APF人工势场奖励（可选）

| 奖励项 | 权重 | 说明 |
|--------|------|------|
| `apf_attractive` | 0.0 | 吸引力势场 (默认关闭) |
| `apf_repulsive` | 0.0 | 排斥力势场 (默认关闭) |

#### 稳定性奖励

| 奖励项 | 权重 | 说明 |
|--------|------|------|
| `height` | 8.0 | 高度跟踪奖励 (目标z=5m) |
| `stability` | 0.05 | 运动稳定性奖励 (线速度+角速度) |

#### 安全惩罚

| 奖励项 | 权重 | 说明 |
|--------|------|------|
| `lidar_threat` | -200.0 | 激光雷达近距离威胁惩罚 |
| `energy` | -0.02 | 能量消耗惩罚（速度+加速度） |
| `action_l2` | -0.005 | 动作L2正则化 |

#### 终端信号

| 奖励项 | 权重 | 说明 |
|--------|------|------|
| `success_bonus` | 800.0 | 到达目标奖励 |
| `collision_penalty` | -800.0 | 碰撞惩罚 |
| `oob_penalty` | -800.0 | 出工作空间惩罚 |
| `timeout_penalty` | -500.0 | 超时惩罚 |

### 4.5 终止条件 (Test6TerminationsCfg)

```python
- time_out:      超时终止 (默认30秒)
- reached_goal:  到达目标 (距离 < 2.5m)
- oob:           出界 (x,y: ±80m, z: 0-10m)
- collision:     碰撞 (接触力 > 1.0)
```

---

## 五、主环境类 (envs/test6_env.py)

### 5.1 MyDroneRLEnv 核心属性

```python
# 目标相关
goal_pos_w: Tensor              # (N, 3) 目标位置世界坐标
_goal_vis_paths: list[str]      # 目标可视化球体的USD路径

# 能量计算缓存
_energy_prev_lin_vel_w: Tensor  # 上一帧线速度
_energy_prev_ang_vel_w: Tensor  # 上一帧角速度

# 进度奖励缓存
_progress_prev_goal_dist: Tensor  # 上一帧目标距离

# Dijkstra导航缓存
_dijkstra_navigator: DijkstraNavigator          # 导航器实例
_dijkstra_distance_fields: Tensor                # (N, H, W) 距离场
_dijkstra_prev_positions: Tensor                 # 上一帧位置
_dijkstra_prev_distances: Tensor                 # 上一帧测地距离
_dijkstra_update_counter: Tensor                 # 更新计数器
_dijkstra_cached_obstacle_positions: Tensor      # 缓存障碍物位置
_dijkstra_occupancy_grid: Tensor                 # 占用栅格

# 元数据
policy_state_dim: int = 17     # 状态维度
policy_lidar_dim: int = 0/432  # 激光雷达维度
```

### 5.2 关键方法

| 方法 | 功能 |
|------|------|
| `_sample_goals(env_ids)` | 随机采样目标位置 (±35m, z=5m) |
| `_refresh_energy_prev_buffers(env_ids)` | 刷新能量计算缓存 |
| `_refresh_progress_prev_dist(env_ids)` | 刷新进度奖励缓存 |
| `_refresh_dijkstra_buffers(env_ids)` | 刷新Dijkstra导航缓存 |
| `_reset_idx(env_ids)` | 环境重置（调用父类+刷新缓存+重新计算观察） |
| `_create_goal_visualizers()` | 创建红色目标可视化球体 |
| `_update_goal_visualizers(env_ids)` | 更新目标可视化位置 |

### 5.3 重置流程 (_reset_idx)

```
1. 调用父类 _reset_idx() → 触发 event:
   - reset_root_state_on_square_edge: 将无人机放到边界，目标放到对侧
   - randomize_obstacles_on_reset: 根据课程等级随机化障碍物

2. 刷新依赖状态的缓存:
   - _refresh_energy_prev_buffers(env_ids)
   - _refresh_progress_prev_dist(env_ids)
   - _refresh_dijkstra_buffers(env_ids)

3. 强制重新计算观察值（因为目标位置已更新）
   obs_dict = self.observation_manager.compute()

4. 返回 (obs_dict, info)
```

---

## 六、MDP模块详解 (tasks/mdp/)

### 6.1 动作定义 (test6_actions.py)

**`RootTwistVelocityActionTerm`**
- 输入: 4D动作 `[-1, 1]^4`
- 映射: `[delta_vx, delta_vy, delta_vz, yaw_rate]` (速度变化量 + 偏航角速度)
- 控制器: LeeVelocityYawRateController
- 输出: 推力(0,0,thrust) + 力矩(torque_x, torque_y, torque_z)

```python
参数:
  vel_scale = 6.0          # 速度变化量缩放 (m/s per step)
  vel_clip = 8.0           # 目标速度限幅
  yaw_rate_scale = 3.14    # 偏航角速度缩放
  yaw_rate_clip = 6.28     # 偏航角速度限幅
  thrust_limit_factor = 3  # 推力限制因子

增量控制逻辑:
  target_vel += delta_v * vel_scale    # 累加速度变化量
  target_vel = clip(target_vel, ±vel_clip)  # 限幅
  v_cmd = target_vel                   # 发送给控制器
```

### 6.2 观察定义 (test6_observations.py)

| 函数 | 输出维度 | 说明 |
|------|----------|------|
| `obs_root_pos_z_norm` | 1D | 高度归一化到[-1,1] |
| `obs_root_quat_norm` | 4D | 四元数归一化 + 半球处理 |
| `obs_root_lin_vel_norm` | 3D | 线速度归一化 |
| `obs_root_ang_vel_norm` | 3D | 角速度归一化 |
| `obs_projected_gravity_norm` | 3D | 投影重力归一化 |
| `obs_goal_delta_norm` | 3D | 目标方向向量归一化 |
| `obs_lidar_min_range_grid` | 432D | 激光雷达最小距离网格 |

**激光雷达网格生成**:
```
参数:
  theta_range: [30°, 90°]    # 俯仰角范围
  phi_range: [0°, 360°]      # 方位角范围
  delta_theta: 10°           # 俯仰角分辨率 (6 bins)
  delta_phi: 5°              # 方位角分辨率 (72 bins)

输出: 6 × 72 = 432D closeness 值 (0=远/空, 1=近)
```

### 6.3 奖励函数详解 (test6_rewards.py)

#### reward_progress_to_goal (进度奖励)

```python
算法:
  d_current = ||goal - pos||          # 当前距离
  d_prev = env._progress_prev_goal_dist  # 上一帧距离
  delta = d_prev - d_current          # 距离减小为正
  towards_speed = delta / dt          # 朝向目标的速度
  reward = clamp(towards_speed / speed_ref, -clip, clip)

  # 更新缓存
  env._progress_prev_goal_dist = d_current
```

#### reward_dijkstra_progress (Dijkstra测地距离奖励)

```python
算法:
  # 1. 检查是否需要更新距离场
  if update_counter >= update_interval:
      # 2. 构建占用栅格 (从障碍物位置)
      occupancy_grid = build_occupancy_grid(obstacle_positions)

      # 3. 从目标位置计算距离场 (wave propagation)
      distance_field = compute_distance_field_fast(occupancy_grid, goal)

      # 4. 更新缓存
      env._dijkstra_distance_fields[env_id] = distance_field

  # 5. 查询当前测地距离
  d_current = get_geodesic_distance(pos, distance_field)
  d_prev = env._dijkstra_prev_distances

  # 6. 计算进度奖励
  delta = d_prev - d_current
  reward = clamp(delta/dt / speed_ref, -clip, clip)

  # 7. 更新缓存
  env._dijkstra_prev_distances = d_current
```

#### penalty_lidar_threat (激光雷达威胁惩罚)

```python
算法:
  # 1. 获取激光雷达点云
  pc = lidar.get_pointcloud(env_ids)  # (N, P, 3)

  # 2. 转换为球坐标
  r = sqrt(x^2 + y^2 + z^2)
  theta = acos(z/r)      # 俯仰角
  phi = atan2(y, x)      # 方位角

  # 3. 分配到网格 bins
  grid = zeros(N, T*P)   # T=6, P=72
  for each point:
      t_idx = (theta - theta_min) / delta_theta
      p_idx = (phi - phi_min) / delta_phi
      bin_idx = t_idx * P + p_idx
      grid[env_id, bin_idx] = min(grid[env_id, bin_idx], r)

  # 4. 计算 closeness
  min_dist = grid.min(dim=1)           # 每个环境最近距离
  closeness = 1.0 - min_dist / max_d   # 0=安全, 1=危险

  # 5. 计算惩罚
  safe_dist = 0.1 * max_d              # 安全距离
  delta = safe_dist - min_dist
  penalty = exp(clamp(delta, 0, inf))  # 指数惩罚
  penalty = clamp(penalty, 0, cap)     # 限幅
```

#### penalty_energy (能量惩罚)

```python
算法:
  v = root_lin_vel_w      # 线速度
  w = root_ang_vel_w      # 角速度

  # 速度惩罚
  pv = (||v|| / lin_vel_scale)^2
  pw = (||w|| / ang_vel_scale)^2

  # 加速度惩罚 (需要缓存)
  a = (v - v_prev) / dt
  alpha = (w - w_prev) / dt
  pa = (||a|| / lin_acc_scale)^2
  palpha = (||alpha|| / ang_acc_scale)^2

  penalty = pv + pw + acc_weight * (pa + palpha)

  # 更新缓存
  env._energy_prev_lin_vel_w = v
  env._energy_prev_ang_vel_w = w
```

### 6.4 终止条件 (test6_terminations.py)

| 函数 | 逻辑 |
|------|------|
| `termination_reached_goal` | `||goal - pos|| < threshold` |
| `termination_out_of_workspace` | `pos < bounds_min or pos > bounds_max` |
| `termination_collision` | `max(contact_force) > threshold` |

### 6.5 事件处理 (test6_events.py)

#### reset_root_state_on_square_edge

```python
功能: 将无人机初始化到正方形边界，目标放到对侧

边界选择 (4条边随机):
  - 左边: x = -35, y ∈ [-35, 35]
  - 右边: x = +35, y ∈ [-35, 35]
  - 底边: y = -35, x ∈ [-35, 35]
  - 顶边: y = +35, x ∈ [-35, 35]

目标位置:
  target_x = -drone_x + noise_x  # 对侧 + 扰动
  target_y = -drone_y + noise_y
  target_z = 5.0
```

#### randomize_obstacles_on_reset

```python
功能: 根据课程等级随机化障碍物位置

输入:
  - current_obstacle_count: 当前应激活的障碍物数量

逻辑:
  1. 将所有障碍物放到 (1000, 1000, -1000) - 隐藏
  2. 对前 N 个障碍物，随机分布在 [-33, 33] × [-33, 33] 区域
  3. 障碍物高度: z = 10/2 = 5m
```

### 6.6 课程学习 (test6_curriculums.py)

#### update_obstacle_curriculum

```python
参数:
  levels = (0, 10, 20, 40, 60, 100)  # 6个难度等级
  success_threshold = 0.8            # 晋级阈值
  window_size = 200                  # 统计窗口

状态变量:
  obstacle_level_idx: int            # 当前等级索引
  obstacle_history: deque(maxlen=200) # 成功历史
  current_obstacle_count: int         # 当前障碍物数量
  obstacle_level_changed: bool        # 等级变化标志

算法:
  1. 从 termination_manager 获取成功信号
  2. 将成功信号加入 history
  3. 计算成功率: success_rate = sum(history) / len(history)
  4. 如果 success_rate >= 0.8 且未达最高级:
       - level_idx += 1
       - current_obstacle_count = levels[level_idx]
       - history.clear()
       - obstacle_level_changed = True
  5. 返回统计信息供TensorBoard记录
```

### 6.7 Dijkstra导航工具 (test6_dijkstra_utils.py)

#### DijkstraNavigator 类

```python
初始化参数:
  grid_size: int = 160          # 栅格尺寸
  cell_size: float = 1.0        # 单元格大小(米)
  workspace_origin: (-80, -80)  # 工作空间原点
  max_distance: 300.0           # 最大距离值

核心方法:
  world_to_grid(pos):           # 世界坐标 → 栅格索引
  grid_to_world(gx, gy):        # 栅格索引 → 世界坐标
  build_occupancy_grid(obstacles):  # 构建占用栅格
  compute_distance_field_fast(occupancy_grid, goal):  # 快速距离场计算
  get_geodesic_distance(pos, distance_field):         # 查询测地距离
```

#### 快速波前传播算法 (compute_distance_field_fast)

```python
算法 (GPU加速):
  1. 初始化距离场: distance_field[goal] = 0, 其他 = max_distance
  2. 创建自由空间掩码: mask = ~occupancy_grid
  3. for iter in range(max_iterations):
       # 使用3x3卷积核计算邻居最小距离
       neighbors = [
           padded[:-2, :-2] + 1.414*cell_size,  # 对角
           padded[:-2, 1:-1] + 1.0*cell_size,   # 上下
           ...  # 8个方向
       ]
       new_distances = min(neighbors, dim=1)

       # 应用掩码和约束
       new_distances = where(mask, new_distances, max_distance)
       new_distances[goal] = 0.0

       # 检查收敛
       if max(abs(current - new_distances)) < 0.01:
           break

       current = new_distances

  4. return current
```

---

## 七、数据流向图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           训练循环 (SKRL)                                │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────────────────┐  │
│  │  Policy │───▶│  Action │───▶│   Env   │───▶│ Next Obs + Reward   │  │
│  │ Network │    │ 4D twist│    │  Step   │    │ + Done + Truncated  │  │
│  └─────────┘    └─────────┘    └────┬────┘    └─────────────────────┘  │
│       ▲                             │                                    │
│       └─────────────────────────────┘                                    │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        MyDroneRLEnv.step()                              │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ 1. ActionTerm.process_actions(): [-1,1]⁴ → [vx,vy,vz,yaw_rate]  │   │
│  │    - 动作裁剪到 [-1, 1]                                          │   │
│  │    - 缩放到实际速度命令                                          │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │ 2. LeeController.apply_actions():                                │   │
│  │    - 积分偏航角: yaw_target += yaw_rate * dt                     │   │
│  │    - 计算推力和力矩                                              │   │
│  │    - 应用到机器人刚体                                            │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │ 3. 物理仿真步进 (IsaacSim)                                       │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │ 4. ObservationManager.compute():                                 │   │
│  │    - 获取机器人状态 (位置、速度、姿态)                           │   │
│  │    - 获取激光雷达点云 → 生成网格                                 │   │
│  │    - 归一化所有观察值                                            │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │ 5. RewardManager.compute():                                      │   │
│  │    - progress_to_goal: 欧氏距离进度                              │   │
│  │    - dijkstra_progress: 测地距离进度                             │   │
│  │    - vel_towards_goal: 朝向目标速度                              │   │
│  │    - lidar_threat: 障碍物威胁惩罚                                │   │
│  │    - energy: 能量消耗惩罚                                        │   │
│  │    - 其他奖励/惩罚项                                             │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │ 6. TerminationManager.compute():                                 │   │
│  │    - 检查超时、到达目标、出界、碰撞                              │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │ 7. CurriculumManager.compute():                                  │   │
│  │    - 更新成功率统计                                              │   │
│  │    - 如达到阈值，提升难度等级                                    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 八、课程学习机制详解

```
┌─────────────────────────────────────────────────────────────┐
│                    障碍物课程学习流程                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  初始化:                                                    │
│    obstacle_level_idx = 0                                   │
│    current_obstacle_count = 0                               │
│    obstacle_history = deque(maxlen=200)                     │
│                                                             │
│  每回合终止时:                                              │
│    ┌─────────────────────────────────────┐                  │
│    │ 获取成功信号 (是否到达目标)          │                  │
│    │ success = termination_reached_goal  │                  │
│    └────────────────┬────────────────────┘                  │
│                     │                                       │
│                     ▼                                       │
│    ┌─────────────────────────────────────┐                  │
│    │ 添加到历史队列                      │                  │
│    │ obstacle_history.append(success)    │                  │
│    └────────────────┬────────────────────┘                  │
│                     │                                       │
│                     ▼                                       │
│    ┌─────────────────────────────────────┐                  │
│    │ 计算成功率                          │                  │
│    │ rate = sum(history) / len(history)  │                  │
│    └────────────────┬────────────────────┘                  │
│                     │                                       │
│                     ▼                                       │
│         ┌───────────────────────┐                           │
│    rate >= 0.8 ?                │                           │
│         │                       │                           │
│        Yes                      No                          │
│         │                       │                           │
│         ▼                       ▼                           │
│    ┌─────────────┐       ┌─────────────┐                    │
│    │ 晋级到下一级 │       │ 保持当前级别 │                    │
│    │ level_idx++ │       │             │                    │
│    │ obstacles = │       │             │                    │
│    │ levels[idx] │       │             │                    │
│    │ history.clear()    │             │                    │
│    │ level_changed=True │             │                    │
│    └─────────────┘       └─────────────┘                    │
│                                                             │
│  下次重置时:                                                │
│    if level_changed:                                        │
│        randomize_obstacles_on_reset()  # 重新生成障碍物      │
│        level_changed = False                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 九、关键文件路径汇总

| 类别 | 文件路径 |
|------|----------|
| **入口脚本** | `omniperception_isaacdrone/scripts/test6_train_skrl.py` |
| **环境类** | `omniperception_isaacdrone/source/omniperception_isaacdrone/envs/test6_env.py` |
| **环境配置** | `omniperception_isaacdrone/source/omniperception_isaacdrone/envs/test6_env_cfg.py` |
| **任务注册** | `omniperception_isaacdrone/source/omniperception_isaacdrone/tasks/test6_registry.py` |
| **动作定义** | `omniperception_isaacdrone/source/omniperception_isaacdrone/tasks/mdp/test6_actions.py` |
| **观察定义** | `omniperception_isaacdrone/source/omniperception_isaacdrone/tasks/mdp/test6_observations.py` |
| **奖励函数** | `omniperception_isaacdrone/source/omniperception_isaacdrone/tasks/mdp/test6_rewards.py` |
| **终止条件** | `omniperception_isaacdrone/source/omniperception_isaacdrone/tasks/mdp/test6_terminations.py` |
| **事件处理** | `omniperception_isaacdrone/source/omniperception_isaacdrone/tasks/mdp/test6_events.py` |
| **课程学习** | `omniperception_isaacdrone/source/omniperception_isaacdrone/tasks/mdp/test6_curriculums.py` |
| **Dijkstra工具** | `omniperception_isaacdrone/source/omniperception_isaacdrone/tasks/mdp/test6_dijkstra_utils.py` |
| **机器人配置** | `omniperception_isaacdrone/source/omniperception_isaacdrone/assets/robots/drone_cfg.py` |
| **激光雷达配置** | `omniperception_isaacdrone/source/omniperception_isaacdrone/assets/sensors/lidar_cfg.py` |
| **Lee控制器** | `omniperception_isaacdrone/source/omniperception_isaacdrone/controller/lee_position_controller.py` |

---

## 十、核心创新点总结

1. **Dijkstra 测地距离奖励**
   - 使用 GPU 加速的波前传播算法
   - 考虑障碍物的最短路径距离
   - 避免局部最优，提供更优的导航信号

2. **全局共享障碍物**
   - 预生成障碍物模板到 `/World/Obstacles`
   - 所有环境共享，提高内存效率
   - 通过位置偏移实现环境间差异

3. **动态课程学习**
   - 基于成功率自动调整难度 (0→100障碍物)
   - 6个难度等级渐进式训练
   - 清空历史避免统计偏差

4. **多目标复合奖励**
   - 进度奖励 + 距离奖励 + 速度奖励
   - Dijkstra奖励 + 稳定性奖励
   - 安全惩罚 (激光雷达 + 能量 + 动作平滑)
   - 终端信号 (成功/碰撞/出界/超时)

5. **对向目标生成**
   - 机器人和目标分别在对侧边界
   - 确保有效导航距离 (~70m)
   - 随机扰动增加多样性

6. **Lee 位置控制器**
   - 将 4D 动作映射到 6D 力/力矩
   - 内部处理偏航角积分
   - 防止负推力保护

---

## 十一、配置参数速查

### 11.1 工作空间参数

```python
WORKSPACE_X = (-80.0, 80.0)    # X轴范围 (米)
WORKSPACE_Y = (-80.0, 80.0)    # Y轴范围 (米)
WORKSPACE_Z = (0.0, 10.0)      # Z轴范围 (米)
GOAL_RADIUS = 2.5              # 目标到达阈值 (米)
square_half_size = 35.0        # 重置边界半尺寸 (米)
```

### 11.2 Dijkstra 参数

```python
grid_size = 80                 # 栅格尺寸 (默认80x80)
cell_size = 2.0                # 单元格大小 (米)
update_interval = 10           # 距离场更新间隔 (步)
speed_ref = 4.0                # 参考速度 (m/s)
clip = 1.0                     # 奖励裁剪值
```

### 11.3 激光雷达参数

```python
theta_range = [30.0, 90.0]     # 俯仰角范围 (度)
phi_range = [0.0, 360.0]       # 方位角范围 (度)
delta_theta = 10.0             # 俯仰角分辨率 (度)
delta_phi = 5.0                # 方位角分辨率 (度)
grid_shape = (6, 72)           # 网格形状
total_bins = 432               # 总维度
max_distance = 50.0            # 最大探测距离 (米)
```

### 11.4 PPO 训练参数

```python
rollouts = 256                 # 每次更新的步数
learning_epochs = 8            # 每次更新的epoch数
mini_batches = 8               # minibatch数量
learning_rate = 1e-4           # 学习率
discount_factor = 0.99         # 折扣因子
lambda = 0.97                  # GAE参数
ratio_clip = 0.2               # PPO裁剪系数
value_clip = 0.2               # 价值函数裁剪
entropy_coef = 1e-2            # 熵奖励系数
```

---

*文档生成时间: 2026-04-07*
*对应代码版本: test6_train_skrl 及相关模块*
