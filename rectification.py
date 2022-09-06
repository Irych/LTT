import cv2
import numpy as np

img1 = cv2.imread('data/input_data/0.png')
img2 = cv2.imread('data/input_data/1.png')

ML = np.array([ [2.7986162739260899e+03, 0., 9.5950000000000000e+02],
                 [0., 2.9851906921878294e+03, 5.3950000000000000e+02],
                 [0., 0., 1. ]])

MR = np.array([[2.8030785191464979e+03, 0., 9.5950000000000000e+02],
               [0., 2.9899504204229311e+03, 5.3950000000000000e+02],
               [0., 0., 1. ]])

DL = np.array([ -3.0195605175804907e-01, 0., 0., 0., 0. ])
DR = np.array([ -2.8760404110746712e-01, 0., 0., 0., 0. ])

img_size = (1920, 1080)
R = np.array([[ 9.9993148262771858e-01, -6.8381673743507926e-03, -9.5010271493625609e-03],
              [6.6700706261733685e-03, 9.9982263176414121e-01, -1.7612926221053014e-02],
       [9.6197821063898990e-03, 1.7548346907522713e-02, 9.9979973760400576e-01 ]])

T = np.array([ -1.0057109534289157e+00, -1.0091959156006969e-02,
       -1.8672493781540369e-02 ])


R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
                        ML,
                        DL,
                        MR,
                        DR,
                        img_size, R, T)


mapx1, mapy1 = cv2.initUndistortRectifyMap(ML, DL, R1, MR,
                                                   img_size,
                                                   cv2.CV_32F)
mapx2, mapy2 = cv2.initUndistortRectifyMap(ML, DL, R2, MR,
                                                   img_size,
                                                   cv2.CV_32F)
img_rect1 = cv2.remap(img1, mapx1, mapy1, cv2.INTER_LINEAR)
img_rect2 = cv2.remap(img2, mapx2, mapy2, cv2.INTER_LINEAR)


total_size = (max(img_rect1.shape[0], img_rect2.shape[0]),
                      img_rect1.shape[1] + img_rect2.shape[1], 3)
img = np.zeros(total_size, dtype=np.uint8)
img[:img_rect1.shape[0], :img_rect1.shape[1]] = img_rect1
img[:img_rect2.shape[0], img_rect1.shape[1]:] = img_rect2

# draw horizontal lines every 25 px accross the side by side image
for i in range(20, img.shape[0], 25):
    cv2.line(img, (0, i), (img.shape[1], i), (255, 0, 0))

cv2.imshow('imgRectified', img)
cv2.imwrite('data/output_data/imgRectified1.png', img_rect1)
cv2.imwrite('data/output_data/imgRectified2.png', img_rect2)
cv2.waitKey()
