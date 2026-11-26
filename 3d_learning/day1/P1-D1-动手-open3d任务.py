# P1-D1 动手任务：用 Open3D 亲手走一遍"表示转换地图"（约 0.6h）
#
# 准备：pip install open3d scipy numpy
# 用法：python P1-D1-动手-open3d任务.py <stage>
#   stage ∈ {a, b, c, d}，建议按顺序做。每个 Stage 末尾有一个思考题，写进笔记。
#
# Stage A (10min): 加载并观察一个点云 —— 建立"点云长什么样"的手感
# Stage B (20min): 深度图 + 内参 K → 反投影成点云 —— 讲义公式的代码版
# Stage C (10min): 点云 → 体素 / 网格 —— 沿转换地图再走两步
# Stage D (10min): 旋转表示实验 —— 亲眼看到万向锁和欧拉角跳变

import sys
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R


# ---------------------------------------------------------------- Stage A
def stage_a():
    """加载示例点云，学会基本观察操作。"""
    ply = o3d.data.PLYPointCloud()          # 自动下载一个示例（客厅场景）
    pcd = o3d.io.read_point_cloud(ply.path)
    print(pcd)                               # 看点数
    print("前5个点:\n", np.asarray(pcd.points)[:5])
    print("边界盒:", pcd.get_axis_aligned_bounding_box())

    # 交互窗口: 鼠标拖转 / 滚轮缩放 / Ctrl+拖动平移 / 按 N 显示法向量
    o3d.visualization.draw_geometries([pcd], window_name="Stage A: 点云")

    # 思考题A: 把 points 数组用 np.random.permutation 打乱行序再显示,
    #          画面会变吗? 为什么? (这就是"无序性"的直观含义)


# ---------------------------------------------------------------- Stage B
def stage_b():
    """深度图 → 点云: z * K^-1 * [u,v,1]^T, 讲义公式的逐行实现。"""
    data = o3d.data.SampleRedwoodRGBDImages()    # 自带 RGBD 示例帧
    color = o3d.io.read_image(data.color_paths[0])
    depth = o3d.io.read_image(data.depth_paths[0])

    depth_np = np.asarray(depth).astype(np.float32) / 1000.0   # mm → m
    H, W = depth_np.shape
    # PrimeSense 默认内参 (先用现成的, 明天 D2 你会亲手标出自己相机的 K)
    fx, fy, cx, cy = 525.0, 525.0, 319.5, 239.5

    # ---- 手写反投影 (不调库, 就是讲义里的公式) ----
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z = depth_np
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    valid = z > 0                             # 深度空洞: 无效像素
    pts = np.stack([x[valid], y[valid], z[valid]], axis=-1)
    print(f"有效点 {len(pts)} / {H*W} 像素 (空洞率 {1 - len(pts)/(H*W):.1%})")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    colors = np.asarray(color)[valid] / 255.0
    pcd.colors = o3d.utility.Vector3dVector(colors)
    # 注意坐标系: OpenCV 光学系 z 朝前 y 朝下, 所以画面是"倒"的 —— 这就是讲义说的轴向约定
    o3d.visualization.draw_geometries([pcd], window_name="Stage B: 手写反投影")

    np.save("/tmp/d1_points.npy", pts)        # 留给 Stage C 用

    # 思考题B: 把 fx 改成 400 再跑一遍, 点云发生了什么形变?
    #          这说明内参错误会造成什么性质的误差? (等比? 径向? 深度方向?)


# ---------------------------------------------------------------- Stage C
def stage_c():
    """点云 → 体素 → 网格: 转换地图上再走两步。"""
    pts = np.load("/tmp/d1_points.npy")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd = pcd.voxel_down_sample(0.01)         # 先降采样, 顺便体会密度不均

    # 点云 → 体素
    voxels = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.03)
    print("体素数:", len(voxels.get_voxels()))
    o3d.visualization.draw_geometries([voxels], window_name="Stage C1: 体素")

    # 点云 → 网格 (Poisson 重建, 需要法向量)
    pcd.estimate_normals()
    pcd.orient_normals_towards_camera_location(np.zeros(3))   # 法向统一朝相机
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh], window_name="Stage C2: Poisson网格")

    # 思考题C: voxel_size 从 0.03 改成 0.01, 体素数变为约几倍? 验证 O(n^3) 直觉。
    #          Poisson 网格在深度空洞处长出了什么? 为什么说"网格有拓扑"既是优点也是风险?


# ---------------------------------------------------------------- Stage D
def stage_d():
    """旋转表示实验: 万向锁 + 欧拉角跳变, 用数字亲眼看到。"""
    print("=== 实验1: 万向锁 ===")
    # pitch=90° 时, yaw 和 roll 塌缩成同一个自由度 (此约定下只有 yaw+roll 的和有意义):
    r1 = R.from_euler("zyx", [30, 90, 0], degrees=True)    # yaw=30, pitch=90, roll=0
    r2 = R.from_euler("zyx", [0, 90, 30], degrees=True)    # yaw=0,  pitch=90, roll=30
    print("r1 与 r2 的欧拉角完全不同, 但旋转矩阵之差:")
    print(np.abs(r1.as_matrix() - r2.as_matrix()).max(), "<- 为0, 是同一个旋转!")

    print("\n=== 实验2: 欧拉角的不连续 ===")
    # 两个物理上几乎相同的旋转, 欧拉角数值差巨大:
    ra = R.from_euler("zyx", [179.5, 10, 20], degrees=True)
    rb = R.from_euler("zyx", [-179.5, 10, 20], degrees=True)
    ang = (ra.inv() * rb).magnitude() * 180 / np.pi
    print(f"实际角度差: {ang:.2f}°, 但 yaw 数值差 359° —— 网络按数值回归会被这里教坏")

    print("\n=== 实验3: 四元数双重覆盖 ===")
    q = R.from_euler("zyx", [40, 20, 10], degrees=True).as_quat()
    print("q  =", np.round(q, 4))
    print("-q =", np.round(-q, 4))
    print("两者是同一旋转:", np.allclose(R.from_quat(q).as_matrix(),
                                       R.from_quat(-q).as_matrix()))

    print("\n=== 实验4: 6D连续表示的构造 (Zhou et al. 2019) ===")
    m = R.from_euler("zyx", [40, 20, 10], degrees=True).as_matrix()
    six_d = m[:, :2].T.flatten()              # 网络输出的就是这 6 个数
    # 从 6D 恢复 R: Gram-Schmidt
    a1, a2 = six_d[:3], six_d[3:]
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - (b1 @ a2) * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)
    m_rec = np.stack([b1, b2, b3], axis=1)
    print("6D 重建误差:", np.abs(m - m_rec).max())

    # 思考题D: 实验2里, 如果损失函数是欧拉角的 L2 距离,
    #          ra 附近的训练样本会把网络往哪个方向拉? 为什么 6D 表示没有这个问题?


if __name__ == "__main__":
    stage = sys.argv[1].lower() if len(sys.argv) > 1 else "a"
    {"a": stage_a, "b": stage_b, "c": stage_c, "d": stage_d}[stage]()
