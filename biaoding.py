import os
import numpy as np
import cv2
import glob


def calib(inter_corner_shape, size_per_grid, img_dir, img_type):
    # criteria: only for subpix calibration, which is not used here.
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    w, h = inter_corner_shape
    # cp_int: corner point in int form, save the coordinate of corner points in world space in 'int' form
    # like (0,0,0), (1,0,0), (2,0,0) ....,(10,7,0).
    cp_int = np.zeros((w * h, 3), np.float32)  # 返回来一个给定形状和类型的用0填充的数组  54行3列
    cp_int[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)  # 三维网格坐标划分  行2列

    # cp_world: corner point in world space, save the coordinate of corner points in world space.
    cp_world = cp_int * size_per_grid

    obj_points = []  # the points in world space
    img_points = []  # the points in image space (relevant to obj_points)
    images = glob.glob(img_dir + os.sep + '**.' + img_type)
    for fname in images:
        img = cv2.imread(fname)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # find the corners, cp_img: corner points in pixel space.
        ret, cp_img = cv2.findChessboardCorners(gray_img, (w, h), None)
        # if ret is True, save.
        if ret == True:
            # cv2.cornerSubPix(gray_img,cp_img,(11,11),(-1,-1),criteria)
            obj_points.append(cp_world)
            img_points.append(cp_img)
            # view the corners
            cv2.drawChessboardCorners(img, (w, h), cp_img, ret)
            cv2.namedWindow('FoundCorners', cv2.WINDOW_NORMAL)
            cv2.resizeWindow("FoundCorners", 600, 600)  # 设置窗口大小
            cv2.imshow('FoundCorners', img)
            cv2.waitKey(1)
    cv2.destroyAllWindows()
    # calibrate the camera
    ret, mat_inter, coff_dis, v_rot, v_trans = cv2.calibrateCamera(obj_points, img_points, gray_img.shape[::-1], None,
                                                                   None)
    #    print(("ret:"), ret)
    print(("internal matrix:\n"), mat_inter)
    # in the form of (k_1,k_2,p_1,p_2,k_3)
    print(("------------------------------------------------------------------"))
    print(("distortion cofficients:\n"), coff_dis)  # 畸变系数k1,k2,k3径向畸变系数, p1,p2是切向畸变系数。
    print(("------------------------------------------------------------------"))
    print(("rotation vectors:\n"), v_rot)
    print(("------------------------------------------------------------------"))
    print(("translation vectors:\n"), v_trans)
    print(("------------------------------------------------------------------"))
    # calculate the error of reproject
    # 反投影误差
    # 通过反投影误差，我们可以来评估结果的好坏。越接近0，说明结果越理想。
    # 通过之前计算的内参数矩阵、畸变系数、旋转矩阵和平移向量，使用cv2.projectPoints()计算三维点到二维图像的投影，
    # 然后计算反投影得到的点与图像上检测到的点的误差，最后计算一个对于所有标定图像的平均误差，这个值就是反投影误差
    total_error = 0
    for i in range(len(obj_points)):
        img_points_repro, _ = cv2.projectPoints(obj_points[i], v_rot[i], v_trans[i], mat_inter, coff_dis)
        error = cv2.norm(img_points[i], img_points_repro, cv2.NORM_L2) / len(img_points_repro)
        total_error += error
    print(("Average Error of Reproject: "), total_error / len(obj_points))
    return mat_inter, coff_dis


def dedistortion(inter_corner_shape, img_dir, img_type, save_dir, mat_inter, coff_dis):
    w, h = inter_corner_shape
    images = glob.glob(img_dir + os.sep + '**.' + img_type)
    for fname in images:
        img_name = fname.split(os.sep)[-1]
        img = cv2.imread(fname)
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mat_inter, coff_dis, (w, h), 0, (w, h))  # 自由比例参数
        dst = cv2.undistort(img, mat_inter, coff_dis, None, newcameramtx)
        # clip the image
        # x,y,w,h = roi
        # dst = dst[y:y+h, x:x+w]
        cv2.imwrite(save_dir + os.sep + img_name, dst)
    print('Dedistorted images have been saved to: %s successfully.' % save_dir)


if __name__ == '__main__':
    inter_corner_shape = (9, 6)
    size_per_grid = 0.026
    img_dir = ".\\pic\\photo5"
    img_type = "jpg"
    # calibrate the camera
    mat_inter, coff_dis = calib(inter_corner_shape, size_per_grid, img_dir, img_type)
    # dedistort and save the dedistortion result.
    save_dir = ".\\pic\\photo5-ch"
    if (not os.path.exists(save_dir)):
        os.makedirs(save_dir)
    dedistortion(inter_corner_shape, img_dir, img_type, save_dir, mat_inter, coff_dis)

