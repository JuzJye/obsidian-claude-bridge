---
type: index
status: active
created: 2026-08-31
updated: 2026-11-13
tags:
  - learning
  - 3d-perception
---

# 3D Learning

[返回首页](../Home.md)

本页只管理学习材料导航；唯一主路线见[3D感知与机器人学习计划](3D感知与机器人学习计划.md)。保留原17周与四个项目，在原内容上新增VLA、世界模型、操作算法，Learning作为扫盲基础；后续完整周约30h是参考预算。

> [!tip] 使用方式
> 先从总体计划确认阶段，再从每日学习直接打开讲义、动手任务和产出。Day 子目录只保存内容，不再维护重复索引。

## 总体计划

| 阶段 | 入口 | 范围 |
| --- | --- | --- |
| 主计划 | [3D感知与机器人学习计划](3D感知与机器人学习计划.md) | 原计划完整保留，含新增三块、时间模板、进度与来源规则 |
| P1 | [第一周扫盲](P1-第一周-扫盲日程.md) | 建立 3D 感知与机器人栈全景 |
| P2 | [深度理解](P2-深度理解-日程.md) | 原几何/深度、检测/6D位姿、重建、视觉触觉/VLM路线，加三块原理 |
| P3 | [项目实战](P3-项目实战-日程.md) | 原基础感知→多模态→RL仿真→多臂/自驾，在各项目内加三块实验 |

## 每日学习

| Day | 主题 | 讲义 | 动手任务 | 产出 |
| --- | --- | --- | --- | --- |
| Day 1 | 3D 表示与坐标系 | [打开讲义](day1/P1-D1-讲义-3D表示与坐标系.md) | [Open3D任务](day1/P1-D1-动手-open3d任务.py) | [3D 表示对比表](day1/3D表示对比表.md) |
| Day 2 | 传感器与深度获取 | [打开讲义](day2/P1-D2-讲义-传感器与深度获取.md) | [标定任务](day2/P1-D2-动手-相机标定任务.py) | [传感器选型决策表](day2/传感器选型决策表.md) |
| Day 3 | 3D任务、Learning、VLA与World Model | [打开讲义](day3/P1-D3-讲义-3D感知任务地图.md) | [五段Python实验](day3/P1-D3-动手-评价指标.py) | [任务-方法-指标总表](day3/3D任务-方法-指标总表.md) |
| Day 4 | 从模仿动作到优化回报、预测后果 | [详细讲义](day4/Daily-Learning-Pack.md) | [CPU最小实验](day4/demo.py) | 五格卡、改变量实验与8道自测；个人验收待完成 |
| Day 5 | 控制、导航、遥操与动作接口 | [详细讲义](day5/Daily-Learning-Pack.md) | [接口与控制实验](day5/demo.py) / [绘图脚本](day5/plot_demo.py) | 动作契约、控制曲线、五篇摘要卡与8道自测；个人验收待完成 |
| Day 6 | VLA与世界模型结构、BEV与环境准备 | [详细讲义](day6/Daily-Learning-Pack.md) | [四组CPU实验](day6/demo.py) / [绘图脚本](day6/plot_demo.py) | 模型结构、预测误差与BEV图、五篇摘要卡与8道自测；个人验收待完成 |
| Day 7 | 数学与工具回查、概念图与P1闭卷验收 | [详细讲义](day7/Daily-Learning-Pack.md) | [数学与仿真实验](day7/demo.py) / [PyTorch实验](day7/torch_demo.py) / [绘图脚本](day7/plot_demo.py) | 36节点概念图、数学推导、三组仿真、五篇摘要卡与10道闭卷题；P1个人验收待完成 |
| Day 8 | 投影、反投影与坐标链，动作空间与TCP | [详细讲义](day8/Daily-Learning-Pack.md) | [CPU实验](day8/demo.py) / [绘图脚本](day8/plot_demo.py) | 坐标链、单位与深度反例、两张图与8道自测；个人验收待完成 |
| Day 9 | 对极几何、三角化与二连杆FK/IK | [详细讲义](day9/Daily-Learning-Pack.md) | [CPU实验](day9/demo.py) / [绘图脚本](day9/plot_demo.py) | E/F与DLT推导、双目误差、IK分支与8道自测；个人验收待完成 |
| Day 10 | PnP、立体匹配、雅可比与阻尼最小二乘 | [详细讲义](day10/Daily-Learning-Pack.md) | [CPU实验](day10/demo.py) / [绘图脚本](day10/plot_demo.py) | PnP噪声对照、视差图、奇异点与DLS对照、8道自测；个人验收待完成 |
| Day 11 | 传统与学习式立体、代价体与Learning基础 | [详细讲义](day11/Daily-Learning-Pack.md) | [CPU实验](day11/demo.py) / [绘图脚本](day11/plot_demo.py) | 匹配比较、梯度训练、场景划分反例、三张图与8道自测；个人验收待完成 |
| Day 12 | 相机标定、畸变、非线性优化与视觉反馈 | [详细讲义](day12/Daily-Learning-Pack.md) | [CPU实验](day12/demo.py) / [绘图脚本](day12/plot_demo.py) | 多视图标定、保留视图检查、尺度与反馈偏置反例、8道自测；个人验收待完成 |
| Day 13 | 几何深度综合实验、GCNet、标定与动作契约 | [详细讲义](day13/Daily-Learning-Pack.md) | [CPU实验](day13/demo.py) / [绘图脚本](day13/plot_demo.py) / [立体与完整标定](day13/stereo_calibration.py) | 深度误差、动作单位与采样周期、GCNet等三篇精读任务、完整标定和10道自测；个人验收待完成 |
| Day 14 | W2几何验收、原论文收束与预训练/微调 | [详细讲义](day14/Daily-Learning-Pack.md) | [CPU实验](day14/demo.py) / [绘图脚本](day14/plot_demo.py) | 几何审计、论文对照、Learning五格卡与参数迁移实验；W2个人验收待完成 |
| Day 15 | PointNet、分类/分割与token表征 | [详细讲义](day15/Daily-Learning-Pack.md) | [PyTorch CPU实验](day15/demo.py) | 手写核心模块、置换检验、关键点与padding反例、8道自测；个人验收待完成 |
| Day 16 | PointNet++、Frustum PointNets与夹爪候选 | [详细讲义](day16/Daily-Learning-Pack.md) | [CPU实验](day16/demo.py) | FPS与邻域、视锥中心偏差、抓取位姿与宽度筛选、8道自测；个人验收待完成 |
| Day 17 | 3D检测评测、配置空间与RRT规划 | [详细讲义](day17/Daily-Learning-Pack.md) | [CPU实验](day17/demo.py) | IoU/AP协议、分数排序对照、RRT/RRT-Connect成功与失败、8道自测；个人验收待完成 |
| Day 18 | CenterNet、2D到3D、BEV与融合、碰撞和轨迹 | [详细讲义](day18/Daily-Learning-Pack.md) | [CPU实验](day18/demo.py) / [绘图脚本](day18/plot_demo.py) / [中心解码与自碰撞](day18/center_self_collision.py) | 表示与误差图、IK/路径/轨迹、CenterNet与两篇补充阅读、9道自测；个人验收待完成 |
| Day 19 | NOCS与6D位姿、旋转与对称、RANSAC/ICP及动作坐标 | [详细讲义](day19/Daily-Learning-Pack.md) | [CPU实验](day19/demo.py) / [绘图脚本](day19/plot_demo.py) | 鲁棒配准、ICP初值、对称与增量动作、三张图与10道自测；个人验收待完成 |
| Day 20 | 检测、投影与位姿综合，ACT与Diffusion Policy | [详细讲义](day20/Daily-Learning-Pack.md) | [CPU实验](day20/demo.py) | 几何链误差、动作chunk与多模态动作、三张图与10道自测；个人验收待完成 |
| Day 21 | 完整感知系统、抓取筛选与W3验收 | [详细讲义](day21/Daily-Learning-Pack.md) | [CPU实验](day21/demo.py) | FoundationPose系统分析、抓取筛选、对比/生成学习、两张图与10道自测；W3个人验收待完成 |
| Day 22 | SfM管线与静态地图、动作条件动力学 | [详细讲义](day22/Daily-Learning-Pack.md) | [CPU实验](day22/demo.py) / [绘图脚本](day22/plot_demo.py) | 匹配与相机恢复、基线误差、运动伪静态重建、两张图与10道自测；个人验收待完成 |
| Day 23 | BA联合优化、部分可观测与潜状态 | [详细讲义](day23/Daily-Learning-Pack.md) | [CPU实验](day23/demo.py) / [绘图脚本](day23/plot_demo.py) | 手写Jacobian/Schur、鲁棒优化、Rome精读、两张图与10道自测；个人验收待完成 |
| Day 24 | NeRF、体渲染与一步动力学 | [详细讲义](day24/Daily-Learning-Pack.md) | [PyTorch CPU实验](day24/demo.py) / [绘图脚本](day24/plot_demo.py) | 小型NeRF训练、保留视角、采样对照、一步预测、三张图与10道自测；个人验收待完成 |
| Day 25 | Instant-NGP、3DGS与多步预测 | [详细讲义](day25/Daily-Learning-Pack.md) | [PyTorch CPU实验](day25/demo.py) / [绘图脚本](day25/plot_demo.py) | 哈希容量、高斯合成、1/5/10步与分布外动作、三张图与10道自测；个人验收待完成 |
| Day 26 | 几何约束重建、碰撞几何与MPC/Dreamer | [详细讲义](day26/Daily-Learning-Pack.md) | [CPU实验](day26/demo.py) | 颜色/深度歧义、碰撞候选评分、两张图与自测；个人验收待完成 |
| Day 27 | 重建、2D/3D特征融合与V-JEPA 2/2-AC | [详细讲义](day27/Daily-Learning-Pack.md) | [CPU实验](day27/demo.py) | 可见性门控、深度噪声、有无动作预测、两张图与自测；个人验收待完成 |
| Day 28 | 统一表示、重建与动态评测、W4验收 | [详细讲义](day28/Daily-Learning-Pack.md) | [CPU实验](day28/demo.py) | 几何完整性、任务关键区域、滚动/OOD误差、两张图与自测；W4个人验收待完成 |
| Day 29 | 视觉触觉与位置、力、阻抗、导纳控制 | [详细讲义](day29/Daily-Learning-Pack.md) | [CPU实验](day29/demo.py) | 接触刚度、稳态与离散稳定性、触觉形变、三张图与自测；个人验收待完成 |
| Day 30 | PBVS/IBVS、接触与滑移纠偏 | [详细讲义](day30/Daily-Learning-Pack.md) | [CPU实验](day30/demo.py) | 交互矩阵、深度和符号误差、滑移与力上限、两张图与自测；个人验收待完成 |
| Day 31 | 触觉到VLM、RT-2与原版OpenVLA动作接口 | [详细讲义](day31/Daily-Learning-Pack.md) | [CPU实验](day31/demo.py) | 动作token、统计量反归一化、坐标与接触门控、两张图与自测；个人验收待完成 |
| Day 32 | pi0与flow matching、BC/DAgger/ACT/Diffusion Policy | [详细讲义](day32/Daily-Learning-Pack.md) | [CPU实验](day32/demo.py) | 连续动作分布、积分方向、多峰合成实验、两张图与自测；个人验收待完成 |
| Day 33 | 操作感知、接触预测与RL/IL数据协议 | [详细讲义](day33/Daily-Learning-Pack.md) | [CPU实验](day33/demo.py) | 接触窗口与概率、阈值、离线支持集、两张图与自测；个人验收待完成 |
| Day 34 | VLM与操作集成、官方VLA条件与LoRA | [详细讲义](day34/Daily-Learning-Pack.md) | [CPU实验](day34/demo.py) | 图文目标选择、RGB-D动作链、冻结与低秩适配、两张图与自测；个人验收待完成 |
| Day 35 | 四组推导、三个核心模块与P2验收 | [详细讲义](day35/Daily-Learning-Pack.md) | [CPU实验](day35/demo.py) / [绘图脚本](day35/plot_demo.py) | 手写PnP、PointNet/BA独立核验、Learning五格卡与精读要求、两张图与10道自测；P2个人验收待完成 |
| Day 36 | 数据集、任务协议与原候选条件 | [详细讲义](day36/Daily-Learning-Pack.md) | [CPU实验](day36/demo.py) | 目标分母、类别/物体级条件与协议实验、实验图与自测；个人验收待完成 |
| Day 37 | 加载器、场景划分与泄漏诊断 | [详细讲义](day37/Daily-Learning-Pack.md) | [CPU实验](day37/demo.py) | 数据契约、分组划分与近邻泄漏反例、实验图与自测；个人验收待完成 |
| Day 38 | GT投影、坐标链与单位检查 | [详细讲义](day38/Daily-Learning-Pack.md) | [CPU实验](day38/demo.py) | 投影真值、裁剪内参、尺度与方向反例、实验图与自测；个人验收待完成 |
| Day 39 | NOCS/EPro-PnP条件与几何基线 | [详细讲义](day39/Daily-Learning-Pack.md) | [CPU实验](day39/demo.py) | 原候选接口、EPnP/RANSAC与坏对应对照、实验图与自测；个人验收待完成 |
| Day 40 | 标准评价、对称性与失败归因 | [详细讲义](day40/Daily-Learning-Pack.md) | [CPU实验](day40/demo.py) | ADD/ADD-S、完整分母与逐目标记录、实验图与自测；个人验收待完成 |
| Day 41 | 观测动作转移契约与时间戳 | [详细讲义](day41/Daily-Learning-Pack.md) | [CPU实验](day41/demo.py) | 曝光/接收时间、动作生效区间与动力学偏差、实验图与自测；个人验收待完成 |
| Day 42 | 数据评估模块文档与W6验收 | [详细讲义](day42/Daily-Learning-Pack.md) | [CPU实验](day42/demo.py) | 审计门禁、错误注入、评价复现与未完成清单、实验图与自测；个人验收待完成 |
| Day 43 | 多尺度方案与密度适应 | [详细讲义](day43/Daily-Learning-Pack.md) | [CPU实验](day43/demo.py) | 局部/全局特征、尺度/稀疏/噪声受控实验、实验图与自测；个人验收待完成 |
| Day 44 | 鲁棒损失、IRLS与位姿求解 | [详细讲义](day44/Daily-Learning-Pack.md) | [CPU实验](day44/demo.py) | Huber/Cauchy、影响函数与带权Kabsch、实验图与自测；个人验收待完成 |
| Day 45 | 几何约束、尺度与深度偏差 | [详细讲义](day45/Daily-Learning-Pack.md) | [CPU实验](day45/demo.py) | SO(3)、规范点与深度约束、低投影误差反例、实验图与自测；个人验收待完成 |
| Day 46 | 五方案受控对照与置信度失效 | [详细讲义](day46/Daily-Learning-Pack.md) | [CPU实验](day46/demo.py) | 固定验证/测试、单项组合、错误sigma压力测试、实验图与自测；个人验收待完成 |
| Day 47 | 抓取候选、宽度与接近质量 | [详细讲义](day47/Daily-Learning-Pack.md) | [CPU实验](day47/demo.py) | 坐标链、接触法向、摩擦/载荷与扰动筛选、实验图与自测；个人验收待完成 |
| Day 48 | IK可达性、碰撞与位姿误差传播 | [详细讲义](day48/Daily-Learning-Pack.md) | [CPU实验](day48/demo.py) | 三关节IK、整条路径检查与捕获代理、实验图与自测；个人验收待完成 |
| Day 49 | 配对消融报告与W7验收 | [详细讲义](day49/Daily-Learning-Pack.md) | [CPU实验](day49/demo.py) | 交互项、场景bootstrap与先验跨域失效、实验图与自测；个人验收待完成 |
| Day 50 | RGB-D输入、标定与跨相机配准 | [详细讲义](day50/Daily-Learning-Pack.md) | [CPU实验](day50/demo.py) | 深度单位、外参/z-buffer、裁剪内参与时间预算、实验图与自测；个人验收待完成 |
| Day 51 | 检测/分割ROI与目标提取 | [详细讲义](day51/Daily-Learning-Pack.md) | [CPU实验](day51/demo.py) | 实例mask、有效深度、污染与目标保留率、实验图与自测；个人验收待完成 |
| Day 52 | 6D位姿与可视化 | [详细讲义](day52/Daily-Learning-Pack.md) | [CPU实验](day52/demo.py) | 物体坐标、鲁棒求解、重投影与三维轴检查、实验图与自测；个人验收待完成 |
| Day 53 | 物体到抓取位姿与IK | [详细讲义](day53/Daily-Learning-Pack.md) | [CPU实验](day53/demo.py) | 三维变换链、TCP/法兰、六关节数值IK、实验图与自测；个人验收待完成 |
| Day 54 | 路径、轨迹与动作日志 | [详细讲义](day54/Daily-Learning-Pack.md) | [CPU实验](day54/demo.py) | 时间参数化、速度加速度、动作契约与回放、实验图与自测；个人验收待完成 |
| Day 55 | RGB-D感知到操作的闭环测试 | [详细讲义](day55/Daily-Learning-Pack.md) | [CPU实验](day55/demo.py) | 观测更新、反馈纠偏、成功分母与失败归因、实验图与自测；个人验收待完成 |
| Day 56 | Π-1可复现交付、系统文档与项目验收 | [详细讲义](day56/Daily-Learning-Pack.md) | [CPU审计实验](day56/demo.py) | 接口误差、完整分母、复现清单与故障注入、实验图与10道自测；正式项目和个人验收待完成 |
| Day 57 | 触觉数据、时钟与在线时间对齐 | [详细讲义](day57/Daily-Learning-Pack.md) | [CPU实验](day57/demo.py) | 仿射时钟、配对mask与到达时间、漂移对照、实验图与自测；个人验收待完成 |
| Day 58 | 触觉特征提取器与参考漂移 | [详细讲义](day58/Daily-Learning-Pack.md) | [CPU实验](day58/demo.py) | 参考差分、统计特征、softmax训练与亮度漂移、实验图与自测；个人验收待完成 |
| Day 59 | 视觉触觉共享表征与对比学习 | [详细讲义](day59/Daily-Learning-Pack.md) | [CPU实验](day59/demo.py) | 双编码器、InfoNCE、错配训练与独立检索、实验图与自测；个人验收待完成 |
| Day 60 | 对齐训练与独立评价 | [详细讲义](day60/Daily-Learning-Pack.md) | [CPU实验](day60/demo.py) | 线性对齐、物体分组、泄漏与错配对照、实验图与自测；个人验收待完成 |
| Day 61 | 接触状态与反馈控制 | [详细讲义](day61/Daily-Learning-Pack.md) | [CPU实验](day61/demo.py) | 接触阶段、受限反馈、延迟与滑移代理、实验图与自测；个人验收待完成 |
| Day 62 | 动作转移、VLA字段与数据边界 | [详细讲义](day62/Daily-Learning-Pack.md) | [CPU实验](day62/demo.py) | 转移契约、有无动作、1/5/10步预测、实验图与自测；个人验收待完成 |
| Day 63 | 编码器对齐与W9验收 | [详细讲义](day63/Daily-Learning-Pack.md) | [CPU实验](day63/demo.py) | 真实梯度训练、独立检索与冻结探针、实验图与自测；个人验收待完成 |
| Day 64 | 融合架构与多模态基线 | [详细讲义](day64/Daily-Learning-Pack.md) | [CPU实验](day64/demo.py) | 早晚融合、质量门控、噪声与模态缺失、实验图与自测；个人验收待完成 |
| Day 65 | 联合编码器与多任务训练 | [详细讲义](day65/Daily-Learning-Pack.md) | [CPU实验](day65/demo.py) | 双分支、任务梯度、验证选择与配对干预、实验图与自测；个人验收待完成 |
| Day 66 | 手写交叉注意与有效信息 | [详细讲义](day66/Daily-Learning-Pack.md) | [CPU实验](day66/demo.py) | Q/K/V、padding与因果mask、全空与置换检查、实验图与自测；个人验收待完成 |
| Day 67 | 6D位姿、对称与接触任务评价 | [详细讲义](day67/Daily-Learning-Pack.md) | [CPU实验](day67/demo.py) | SO(3)、ADD/MSSD、概率校准与完整分母、实验图与自测；个人验收待完成 |
| Day 68 | 多模态动作条件预测 | [详细讲义](day68/Daily-Learning-Pack.md) | [CPU实验](day68/demo.py) | 残差转移、动作扫描、接触分层与触觉偏置、实验图与自测；个人验收待完成 |
| Day 69 | 闭合状态、多步预测与触觉/深度消融 | [详细讲义](day69/Daily-Learning-Pack.md) | [CPU实验](day69/demo.py) | 历史辨识、1/5/10步与自由展开、四组消融/偏置对照、四张图与10道自测；个人验收待完成 |
| Day 70 | 融合与动态双任务报告、W10验收 | [详细讲义](day70/Daily-Learning-Pack.md) | [CPU配对评价](day70/demo.py) | 单位与分母、episode bootstrap、删难例对照、四张图与10道自测；正式项目和个人验收待完成 |

新增 `dayN/` 后，只需在“每日学习”表中增加一行。以上内部链接同时构成 Obsidian 关系图谱中的连接。

2026-10-29检查时，Day3原交互HTML及说明页未在本地找到；本页先保留已存在的讲义、实验和产出入口。

## 当前阅读顺序

**近期材料按原顺序阅读：** [Day 57](day57/Daily-Learning-Pack.md) → [Day 58](day58/Daily-Learning-Pack.md) → [Day 59](day59/Daily-Learning-Pack.md) → [Day 60](day60/Daily-Learning-Pack.md) → [Day 61](day61/Daily-Learning-Pack.md) → [Day 62](day62/Daily-Learning-Pack.md) → [Day 63](day63/Daily-Learning-Pack.md) → [Day 64](day64/Daily-Learning-Pack.md) → [Day 65](day65/Daily-Learning-Pack.md) → [Day 66](day66/Daily-Learning-Pack.md) → [Day 67](day67/Daily-Learning-Pack.md) → [Day 68](day68/Daily-Learning-Pack.md) → [Day 69](day69/Daily-Learning-Pack.md) → [Day 70](day70/Daily-Learning-Pack.md)。若个人学习仍在更早课程，从上表对应位置继续；材料归档位置不等于个人学习位置。不因补写重排课程，也不以后一课的前置摘要替代缺课。

**2026-09-22补齐记录：** Day 8–12、Day 14–17的九份讲义、配套代码与实验图已补齐。Day 13补足原定GCNet、立体方法比较和完整标定实验；Day 18补足原定Objects as Points / CenterNet和自碰撞检查。补写或修订日期单独记录，原课程日期保留在各讲义的`scheduled_date`中。

**2026-09-23当日材料：** [Day 19](day19/Daily-Learning-Pack.md)已归档，含NOCS与6D位姿、旋转与对称、RANSAC/ICP、绝对和增量动作；配套CPU实验及三张图已验证。补齐Day 8–18后按顺序衔接本课。

**2026-09-29补齐记录：** Day 20–25共六课已写入，覆盖9月24–28日缺失材料和9月29日当日材料，完整保留原定主题、课号及课程日期。包含详细讲义、配套CPU实验、15张实验图与文末自测答案；代码已实际执行，讲义与结果已核对。官方大型模型完整复现、真实设备执行与个人学习验收仍按各课要求独立记录。

**2026-10-09补齐记录：** Day 26–35共十课已写入，原定课程日期依次保留2026-09-30至2026-10-09，实际创建、修订和CPU验证日期为2026-10-09。覆盖W4重建/世界模型收束及W5视觉触觉/VLM/操作感知，含完整讲义、配套实验、21张图与文末自测答案。Day35补足手写非共面DLT-PnP与阻尼精化，并独立核验已有PointNet和BA模块。20篇精读要求是验收清单，未替用户勾选阅读完成；官方大模型完整复现、官方VLA权重推理、真机执行与个人学习验收仍待各自证据。

**2026-10-24补齐记录：** Day36–50十五课已归档，原定日期依次为2026-10-10至10-24，实际创建、修订与CPU验证为2026-10-24。包含完整讲义、15个独立脚本、28张已查看实验图、单变量对照和文末答案；分别记录合成验证、官方基线/真实数据、设备执行与个人验收，未把代理实验写成官方网络复现。

**2026-10-29补齐记录：** Day51–55五课已按原日序写入，scheduled_date保留2026-10-25至10-29，实际创建/修订/CPU验证为2026-10-29。包含ROI与点云提取、6D位姿可视化、物体到抓取/IK、轨迹动作日志、闭环测试；五个独立教学脚本、10张主实验图和单变量结果已检查。现有环境未安装MuJoCo/PyBullet，Day55完成的是明确标注的合成几何/控制闭环，原计划的真实RGB-D或高质量接触物理仿真正式验收仍待独立证据。资料生成、代码验证和个人学习保持分开。

**2026-10-30当日材料：** [Day56](day56/Daily-Learning-Pack.md)已归档，包含Π-1可复现交付、系统接口与误差预算、完整分母、缺失结果界限、Wilson算例、系统文档和演示证据。独立CPU审计脚本、默认与删除失败行对照、四张图和另外四种错误注入已检查。40条结果为预设合成台账，不是新增物理试验；正式基准、真实/接触物理闭环和个人验收仍待各自证据。

**2026-11-11补齐记录：** Day57–68共十二课已按原顺序归档，scheduled_date保留2026-10-31至2026-11-11，实际创建、修订与CPU验证日期为2026-11-11。覆盖W9触觉数据/特征/对齐学习/独立评价/接触控制/动作转移/周验收，以及W10融合基线/联合编码/交叉注意/位姿接触评价/动作条件预测。十二份详细讲义、独立CPU脚本、46张默认及对照图、单变量实验与文末答案已检查；完整嵌入代码、日期和相对链接已核对。D59/D63错配训练使用独立随机流，避免同时改变模型初始化。D64的强噪声例与D68的触觉偏置例保留融合失效证据，不预设多模态一定更好。全部是明确标注的合成教学数据或算子/评价实验，正式真实数据、接触物理、官方模型复现及个人验收仍需独立证据。

**2026-11-12当日材料：** [Day69](day69/Daily-Learning-Pack.md)已归档，包含闭合状态、历史刚度辨识、自由rollout、误差传播、1/5/10步端点与整段评价、触觉/深度四组消融。独立CPU脚本、默认及测力+0.2 N对照、四张图和10道自测已检查。默认完整组合第10步位置RMSE1.108541 mm，偏置后3.059704 mm，劣于无触觉V+D的2.360731 mm；域外材料中偏置局部改善，保留误差抵消的边界。实验为已知解析动力学加历史参数辨识，不复用D68网络；正式预测不读取未来观测或真实刚度，特权重置诊断独立标明。真实设备、复杂接触与个人验收仍待独立证据。

**2026-11-13当日材料：** [Day70](day70/Daily-Learning-Pack.md)已归档，分别报告当前融合任务与动态预测，包含指标单位、配对episode bootstrap、缺失分母、多目标判断、W10证据矩阵和10道自测。独立CPU脚本、默认/删除8个难例对照与四张图已检查；48个合成episode上，三维平移RMSE为6.630561→1.922090 mm，未来x@10为3.276268→1.427360 mm。删记录后表面误差下降，但正式fusion、差值和CI均为null；固定估计器和已知弹簧方程没有网络训练或机器人尝试。资料、代码、正式项目与个人验收分别记录，W11按原日序进入语言查询。

日历锚点仍为2026-09-08→Day 4，累计delay为0；截至2026-11-13，Day1–70课程资料无缺口。下一门新课为2026-11-14的Day71 / W11-D1 / P3项目Π-2：CLIP或VLM语言查询；Π-1正式项目未完成项继续独立保留，Π-2正式项目与个人验收尚待证据，资料归档位置不等于个人学习位置。生成及漏课补写规则见[唯一主计划](3D感知与机器人学习计划.md)。

讲义沿用Day3的知识组织与深度，不设阅读时间分档；每天北京时间09:00开始生成，完成后消息只给简报与入口。D1、D2的原讲义、个人批改记录和已找到的配套脚本保留。资料与代码验证不代表个人完成学习，各阶段是否通过以实际作答、个人产出和验收为准。
