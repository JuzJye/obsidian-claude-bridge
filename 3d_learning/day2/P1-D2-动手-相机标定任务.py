# P1-D2 动手任务：棋盘格标定, 亲手看清 K 和 dist (约 0.4h)
#
# 准备：pip install opencv-python numpy
# 两种模式:
#   模式1 (推荐, 用你的 D405): 拍 15-20 张棋盘格照片放进一个文件夹
#       python P1-D2-动手-相机标定任务.py ./my_calib_imgs 9 6 25.0
#       (参数: 图片目录 / 内角点列数 / 行数 / 方格边长mm)
#       拍摄要诀: 覆盖视场四角和边缘 | 近中远都要有 | 棋盘倾斜各方向 30° 左右都拍
#   模式2 (无相机时): 不带参数运行, 自动下载 OpenCV 官方样例图 (9x6 棋盘)
#       python P1-D2-动手-相机标定任务.py
#
# 跑完后回答脚本末尾的思考题, 写进选型决策表笔记。

import sys
import glob
import os
import urllib.request
import numpy as np
import cv2

SAMPLE_DIR = "/tmp/d2_calib_samples"
SAMPLE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/{}"


def get_images():
    if len(sys.argv) >= 2:
        img_dir, cols, rows = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
        square = float(sys.argv[4]) if len(sys.argv) > 4 else 25.0
        paths = sorted(sum([glob.glob(os.path.join(img_dir, e))
                            for e in ("*.jpg", "*.png", "*.bmp")], []))
    else:
        print("未指定目录, 下载 OpenCV 样例图 (9x6 棋盘, 方格约 25mm) ...")
        os.makedirs(SAMPLE_DIR, exist_ok=True)
        paths = []
        for i in range(1, 15):
            name = f"left{i:02d}.jpg"
            p = os.path.join(SAMPLE_DIR, name)
            if not os.path.exists(p):
                try:
                    urllib.request.urlretrieve(SAMPLE_URL.format(name), p)
                except Exception:
                    continue
            if os.path.exists(p):
                paths.append(p)
        cols, rows, square = 9, 6, 25.0
    assert len(paths) >= 8, f"只有 {len(paths)} 张图, 至少要 8 张 (推荐 15+)"
    return paths, cols, rows, square


def main():
    paths, cols, rows, square = get_images()
    pattern = (cols, rows)

    # 世界坐标: 棋盘自己的坐标系, z=0 平面上的格点 (这就是"已知世界点")
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square

    obj_pts, img_pts, used, size = [], [], [], None
    for p in paths:
        img = cv2.imread(p)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        ok, corners = cv2.findChessboardCorners(gray, pattern, None)
        if not ok:
            print(f"  [跳过] 角点检测失败: {os.path.basename(p)}")
            continue
        # 亚像素精化 —— 讲义里"0.1px 匹配精度"的近亲
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3))
        obj_pts.append(objp); img_pts.append(corners); used.append(p)
    print(f"有效图片 {len(used)}/{len(paths)}")

    # ---- 标定: 就是讲义里那个最小化重投影误差的非线性最小二乘 ----
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_pts, img_pts, size, None, None)

    np.set_printoptions(precision=2, suppress=True)
    print("\n=== 标定结果 ===")
    print(f"整体 RMS 重投影误差: {rms:.3f} px  ({'合格 <0.5' if rms < 0.5 else '偏大, 检查拍摄覆盖'})")
    print("K =\n", K)
    print(f"fx={K[0,0]:.1f} fy={K[1,1]:.1f} (差异 {abs(K[0,0]-K[1,1])/K[0,0]:.2%})")
    print(f"主点 c=({K[0,2]:.1f}, {K[1,2]:.1f}), 图像中心=({size[0]/2}, {size[1]/2})")
    print("畸变 [k1 k2 p1 p2 k3] =", dist.ravel())
    print("  k1 %s0 → %s形畸变" % (("<", "桶") if dist[0, 0] < 0 else (">", "枕")))

    # ---- 每张图的误差: 找出拍坏的那张 ----
    print("\n=== 逐图重投影误差 (误差最大的那张值得回看拍摄姿态) ===")
    for i, p in enumerate(used):
        proj, _ = cv2.projectPoints(obj_pts[i], rvecs[i], tvecs[i], K, dist)
        e = cv2.norm(img_pts[i], proj, cv2.NORM_L2) / len(proj)
        print(f"  {os.path.basename(p):20s} {e:.3f} px")

    # ---- 角点覆盖图: 比 RMS 数字更重要的体检 ----
    cover = np.zeros((size[1], size[0]), np.uint8)
    for c in img_pts:
        for pt in c.reshape(-1, 2).astype(int):
            cv2.circle(cover, tuple(pt), 8, 255, -1)
    cv2.imwrite("/tmp/d2_coverage.png", cover)
    ratio = (cover > 0).mean()
    print(f"\n角点覆盖率 {ratio:.1%} (已存 /tmp/d2_coverage.png, 边角发黑=畸变系数不可信)")

    # ---- 去畸变前后对比 ----
    img = cv2.imread(used[0])
    und = cv2.undistort(img, K, dist)
    cv2.imwrite("/tmp/d2_undistort_compare.png", np.hstack([img, und]))
    print("去畸变对比已存 /tmp/d2_undistort_compare.png (盯着图里的直线看)")

    # 思考题1: fx 和 fy 差多少? 主点偏离中心几个像素? 对照讲义解释原因。
    # 思考题2: 打开覆盖图, 视场边缘有没有黑区? 如果有, k2/k3 的数值还可信吗?
    # 思考题3 (有 D405 时): 用 pyrealsense2 读 EEPROM 内参 (你写在 D1 讲义里的代码),
    #          和自己标的 K 对比, fx 差百分之几? 这个差异经反投影会造成多大的点云误差?
    #          (提示: 用昨天思考题B的结论)


if __name__ == "__main__":
    main()
