import numpy as np
import math

def object_point_world_position(u, v, w, h, p, k):
    u1 = u
    v1 = v + h / 2
    # alpha = -(90 + 0) / (2 * math.pi)
    # peta = 0
    # gama = -90 / (2 * math.pi)

    fx = k[0, 0]
    fy = k[1, 1]
    # 相机高度
    # 关键参数，不准会导致结果不对
    H = 740
    # 相机与水平线夹角, 默认为0 相机镜头正对前方，无倾斜
    # 关键参数，不准会导致结果不对
    angle_a = 0
    angle_b = math.atan((v1 - H / 2) / fy)
    angle_c = angle_b + angle_a


    depth = (H / np.sin(angle_c)) * math.cos(angle_b)


    k_inv = np.linalg.inv(k)
    p_inv = np.linalg.inv(p)
    # print(p_inv)
    point_c = np.array([u1, v1, 1])
    point_c = np.transpose(point_c)

    # 相机坐标系下的关键点位置
    c_position = np.matmul(k_inv, depth * point_c)

    # 世界坐标系下
    c_position = np.append(c_position, 1)
    c_position = np.transpose(c_position)
    c_position = np.matmul(p_inv, c_position)
    d1 = np.array((c_position[0], c_position[1]), dtype=float)
    distance = math.sqrt(math.pow(d1[0], 2) + math.pow(d1[1], 2))
    return distance/1000
