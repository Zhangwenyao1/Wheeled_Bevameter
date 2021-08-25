import pyrealsense2 as rs
import numpy as np
import cv2
import time

import numpy as np
import cv2.aruco as aruco
import matplotlib.pyplot as plt
import math
import csv
from sklearn import linear_model,datasets
import apriltag

def getPoint(cx,cy,r,stx,sty,edx,edy):
    k = (edy - sty) / (edx - stx);
    b = edy - k*edx;
    c = cx*cx + (b - cy)*(b- cy) -r*r;
    a = (1 + k*k);
    b1 = (2*cx - 2*k*(b - cy));

    tmp = math.sqrt(b1*b1 - 4*a*c);
    x1 = ( b1 + tmp )/(2*a);
    y1 = k*x1 + b;
    x2 = ( b1 - tmp)/(2*a);
    y2 = k*x2 + b;
# 判断求出的点是否在圆上
    res1 = (x1 -cx)*(x1 -cx) + (y1 - cy)*(y1 -cy)
    res2 = (x2 - cx) * (x2 - cx) + (y2 - cy) * (y2 - cy)
    p_cirleline1 =np.array(([0,0]))
    p_cirleline2 =np.array(([0,0]))
    if( r*r-0.01<res1< r*r+0.01):
        p_cirleline1[0] = x1
        p_cirleline1[1] = y1
    if(r*r-0.01<res2< r*r+0.01):
        p_cirleline2[0] = x2
        p_cirleline2[1] = y2
    return p_cirleline1,p_cirleline2

# 测试


    '''求两向量的夹角 参考网址 ：https://blog.csdn.net/qq_28382661/article/details/109379842'''
def angle(vec1, vec2, deg=False):
    _angle = np.arctan2(np.abs(np.cross(vec1, vec2)), np.dot(vec1, vec2))
    if deg:
        _angle = np.rad2deg(_angle)
    return _angle
'''求两点间的距离，自己写的'''
def distanceofpint(point0,point1):
    if point0.shape[0] == 3:
        return ((point0[0] - point1[0])**2 + (point0[1] - point1[1])**2 + (point0[2] - point1[2])**2)**0.5
    if point0.shape[0] == 2:
        return ((point0[0] - point1[0])**2 + (point0[1] - point1[1])**2)**0.5
'''最小二乘法拟合曲线，参考网址：https://blog.csdn.net/m0_38128647/article/details/75689228'''
def Least_squares(x,y):
    x_ = x.mean()
    y_ = y.mean()
    m = np.zeros(1)
    n = np.zeros(1)
    k = np.zeros(1)
    p = np.zeros(1)
    lenx = len(x)
    for i in np.arange(0,lenx):
        k = (x[i]-x_)* (y[i]-y_)
        m += k
        p = np.square( x[i]-x_ )
        n = n + p
    a = m/n
    b = y_ - a* x_
    return a,b

'''点到直线的距离，参考网址：https://blog.csdn.net/sinat_29957455/article/details/107490561'''
def get_distance_from_point_to_line(point, line_point1, line_point2):
    # 对于两点坐标为同一点时,返回点与点的距离
    if line_point1 == line_point2:
        point_array = np.array(point )
        point1_array = np.array(line_point1)
        return np.linalg.norm(point_array -point1_array )
    # 计算直线的三个参数
    A = line_point2[1] - line_point1[1]
    B = line_point1[0] - line_point2[0]
    C = (line_point1[1] - line_point2[1]) * line_point1[0] + \
        (line_point2[0] - line_point1[0]) * line_point1[1]
    # 根据点到直线的距离公式计算距离
    distance = np.abs(A * point[0] + B * point[1] + C) / (np.sqrt(A**2 + B**2))
    return distance

def detect():
    corner = [0,0,0,0,0,0,0,0,0,0,0,0,0]
    pipeline = rs.pipeline( )
    config = rs.config( )
    pipeline.start(config)
    time.sleep(1)
    index = 1
    sinkage = 0
    length = 0.0268
    try :
        while (index<4):

            
            frames = pipeline.wait_for_frames( )
            depth_frame = frames.get_depth_frame( )
            color_frame = frames.get_color_frame( )
            if not depth_frame or not color_frame :
                continue
            # depth_image = np.asanyarray(depth_frame.get_data( ))
            color_image = np.asanyarray(color_frame.get_data( ))
            # color_image = cv2.imread('/home/zwy/Desktop/DynamixelSDK-3.7.31/python/tests/protocol2_0/zwy_Color.png')
            # print(color_image.shape)
            # img = cv2.GaussianBlur(color_image, (5, 5), 0)  # Gaussian Filter process
            image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
            img_H = image[:, :, 0]
            img_S = image[:, :, 1].copy( )
            img_I = image[:, :, 2]
            h, w, c = color_image.shape

            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            
            '''形态学'''
            (thresh2, im_bw2) = cv2.threshold(img_S, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            cv2.imshow('otsu2', color_image)
            cv2.waitKey(1)
            structure_element = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
            # img_close = cv2.morphologyEx(im_bw2, cv2.MORPH_CLOSE, structure_element)
            # cv2.imshow('Closing', img_close)  # 闭运算得出的图片
            img_open = cv2.morphologyEx(im_bw2, cv2.MORPH_OPEN, structure_element)
            # cv2.imshow('Opening ', img_open)  # 开运算得出的图片
            # cv2.waitKey(0)
            '''边缘提取算法，此处使用Canny算子'''
            edges = cv2.Canny(img_open, 50, 200)  # 参数:图片，minval，maxval,kernel = 3
            # cv2.imshow('edges', gray)

            '''Hough-circle transform，但是效果并不好'''  # 利用分段线性算法来提高
            # circles = cv2.HoughCircles(im_bw2, cv2.HOUGH_GRADIENT, 4, 2000, param1=90, param2=70, minRadius=100, maxRadius=300)
            # circles = circles[0, :, :]   #提取为二维
            # circles = np.uint16(np.around(circles))  # 四舍五入，取整
            # result = edges.copy( )
            # lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100,maxLineGap=10)
            # for line in lines :
            #     x1, y1, x2, y2 = line[0]
            #     print(x1,y1,x2,y2)
            #     cv2.line(color_image, (x1, y1), (x2, y2), (255, 255, 255), 2)

            # '''求解圆心，使用opencv arcuo 算法'''
            img = color_image
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        # color_image = cv2.imread('123_Color.png')
        # print(color_image.shape)
        # img = cv2.GaussianBlur(color_image, (5, 5), 0)  # Gaussian Filter process
        # 创建一个apriltag检测器
            at_detector = apriltag.Detector(apriltag.DetectorOptions(families='tag36h11 tag25h9') )
            # at_detector = apriltag.Detector(families='tag36h11 tag25h9')  #for windows
        # 进行apriltag检测，得到检测到的apriltag的列表
            tags = at_detector.detect(gray)
            # # 使用arcuo自带算法找到二维码的角点

            '''------------------求两点间的真实距离----------------------'''
            for tag in tags:
                cv2.circle(img, tuple(tag.corners[0].astype(int)), 4,(255,0,0), 2) # left-top
                cv2.circle(img, tuple(tag.corners[1].astype(int)), 4,(255,0,0), 2) # right-top
                cv2.circle(img, tuple(tag.corners[2].astype(int)), 4,(255,0,0), 2) # right-bottom
                cv2.circle(img, tuple(tag.corners[3].astype(int)), 4,(255,0,0), 2) # left-bottom
                center = np.array(
                [int((tag.corners[0][0] + tag.corners[2][0]) / 2), int((tag.corners[0][1] + tag.corners[2][1]) / 2)]).reshape(2, 1)
                center1 = np.array([(tag.corners[0][0] + tag.corners[2][0]) / 2, (tag.corners[0][1] + tag.corners[2][1]) / 2])
                cv2.circle(img, tuple(center.astype(float)), 6, (0, 255, 255), 2)  # circle center
                # print('center of img', center)


                if (tag.corners[1]).shape == (2,) :
                    pc1 = np.append(tag.corners[1], 1).reshape(3, 1)
                    pc2 = np.append(tag.corners[2], 1).reshape(3, 1)
                    # zw1 = tvecs[2].squeeze( )
                    # point0 = np.dot(R_1, ((np.dot(K_1, pc1) * zw1 - tvecs)))
                    # point1 = np.dot(R_1, ((np.dot(K_1, pc2) * zw1 - tvecs)))
                    # print('point0',point0)
                    # print('point1',point1)
                    diameter = distanceofpint(pc1, pc2) / 0.0268 * 0.0952 # 小车轮
                    # diameter = distanceofpint(pc1, pc2) / 0.0268 * 0.141

                    yuan = cv2.circle(img, tuple(center.astype(float)), int(diameter / 2), (0, 255, 255),
                                      2)  # circle
                    print('center of img', center)

                    DST1 = (distanceofpint(tag.corners[0], center) + distanceofpint(tag.corners[1], center) + distanceofpint(
                    tag.corners[2], center) + distanceofpint(tag.corners[3], center)) / 4
                    DST = 4/2.68 * DST1 # 大正方形的半径为小正方形半径的2倍

                    point2 = np.array([center[0]-DST,center[1]])
                    cv2.circle(img, tuple(center.astype(float)), int(DST), (255, 255, 255), 2)

                    # cv2.imshow('otsu2', img)

                # for i in circles[:] :
                #     cv2.circle(color_image, (i[0], i[1]), i[2], (255, 0, 0), 5)  # 画圆
                #     cv2.circle(color_image, (i[0], i[1]), 2, (255, 0, 0), 10)  # 画圆心
                #     print(i[0],i[1])

                '''由于二位码边缘会被被边缘提取算法提取出来影响效果，此处将正方形轮廓变为0'''

                for i in range(int(center[0] - DST-10), int(center[0] + DST+10)) :
                    for j in range(int(center[1] - DST-10), int(center[1] + DST+10)) :
                        if ((i - center[0])**2 + (j - center[1])**2)**0.5 <= DST + 20 :
                            edges[j, i] = 0

                ''' 使用稳健回归算法来拟合沉陷曲线'''
                x = []
                y = []
                radius = diameter / 2  # 图像中的轮子半径

                ### 因为在图像处理过程中有一些异常点，所以应该缩小半径来移除这些异常点
                for i in range(int(0), int(center[0])) :
                    for j in range(0, int(h - 1)) :
                        if (((i - center[0])**2 + (j - center[1])**2) <= radius**2) and (edges[j, i] > 10) :
                            edges[j, i] = 250
                            x.append(i)
                            y.append(j)
                edges[edges > 250] = 0
                x = np.array(x)
                y = np.array(y)




                #
                if len(x) < 2:
                    continue
                else:
                    index +=1
                    model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression( ),
                                                                max_trials=100,  # 最大迭代次数
                                                                min_samples=30,  # 最小抽取的内点样本数量
                                                                )
                    model_ransac.fit(x.reshape(-1, 1), y)
                    # print("RANSAC算法参数： ", model_ransac.estimator_.coef_)

                    # '''此处使用了最小二乘法，但发现偏差过大'''
                    # a,b = Least_squares(x,y)
                    # # print (a,b)
                    # y1 = a * x + b
                    '''------------------------最小二乘法---------------------------'''

                    y2 = model_ransac.estimator_.coef_ * x + model_ransac.estimator_.intercept_
                    a = model_ransac.estimator_.coef_  # 直线斜率
                    b = model_ransac.estimator_.intercept_  # 截距
                    # plt.figure('最小二乘',figsize=(10, 5), facecolor='w')
                    # plt.plot(x, y, 'bo', lw=2, markersize=6)
                    # plt.plot(x, y1, '-', lw=2, markersize=6)
                    # plt.grid(b=True, ls=':')
                    # plt.xlabel(u'X', fontsize=16)
                    # plt.ylabel(u'Y', fontsize=16)
                    # plt.show()
                    # plt.figure('稳健回归',figsize=(10, 5), facecolor='w')
                    # plt.plot(x, y, 'bo', lw=2, markersize=6)
                    # plt.plot(x, y2, 'g', lw=2, markersize=6)
                    # plt.grid(b=True, ls=':')
                    # plt.xlabel(u'X', fontsize=16)
                    # plt.ylabel(u'Y', fontsize=16)
                    # plt.show()
                    p1 = np.array((0, 0))
                    p2 = np.array((0, 0))
                    p1[0] = (min(y) - b) / a
                    p1[1] = min(y)
                    p2[0] = (max(y) - b) / a
                    p2[1] = max(y)
                    # p1 = ((min(y)-b)/a,min(y))
                    # p2 = ((max(y)-b)/a,max(y))

                    cross_point1, cross_point2 = getPoint(center1[0], center1[1], radius, p1[0], p1[1], p2[0], p2[1])

                    ''' 求出沉陷量'''
                    # distance = get_distance_from_point_to_line(center,((min(y)-b)/a,min(y)),((max(y)-b)/a,max(y)))
                    distance = get_distance_from_point_to_line(center, tuple(p1), tuple(p2))
                    # radius = distanceofpint(pc1, pc2) / length * 0.096/2
                    # radius = distanceofpint(pc1, pc2) / length * 0.1/2
                    radius = DST1/(length/2 * (2**0.5))* 0.09556/2
                    sink = length/2 * (2**0.5) / DST1 * (radius - distance)
                    print('沉陷量', sink)
                    sinkage += sink
                    print('sinkage',sinkage)
                    # # '''求进入角和退出角'''
                    # cross_point = np.array([0.0, 0.0])
                    # cross_point[0] = (center1[1] - b) / a
                    # cross_point[1] = center1[1]
                    # vec1 = center1 - cross_point1
                    # vec2 = center1 - cross_point2
                    # # cv2.line(color_image, tuple(p1), tuple(center1.astype(int)), (255, 255, 255), 2)
                    # cv2.line(color_image, tuple(cross_point1.astype(int)), tuple(center1.astype(int)), (255, 255, 255), 2)
                    # cv2.line(color_image, tuple(cross_point2.astype(int)), tuple(center1.astype(int)), (255, 255, 255), 2)
                    # cv2.line(color_image, tuple(cross_point1.astype(int)), tuple(cross_point2.astype(int)), (255, 255, 255),
                    #          2)
                    # # vec3 = center1 - cross_point
                    # # angle1 = angle(vec1, vec3)
                    # # angle1 = angle1 * 360 / 2 / np.pi
                    # # angle2 = angle(vec2, vec3)
                    # # angle2 = angle2 * 360 / 2 / np.pi
                    # # print(angle1)
                    # # print(angle2)

            # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('edges', edges)
            cv2.imshow('RealSense', color_image)
            key = cv2.waitKey(0)
            # if key & 0xFF == ord('q') or key == 27:
            #     cv2.destroyAllWindows()
            #     break
                    # cv2.imshow('edges', edges)
                    # key = cv2.waitKey(0)
                    # cv2.waitKey(0)
        # print('平均沉陷量',sinkage/3)

    
    finally :
    #     # Stop streaming
        pipeline.stop( )
    return sinkage/3


if __name__ == "__main__":
    detect()