from __future__ import print_function

import numpy as np
import cv2 as cv

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')


def main():
    print('loading images...')
    imgL = cv.imread('data/output_data/imgRectified1.png')  # downscale images for faster processing
    imgR = cv.imread('data/output_data/imgRectified2.png')

    # disparity range is tuned for 'aloe' image pair
    window_size = 3
    min_disp = 0
    num_disp = 256
    stereo = cv.StereoSGBM_create(minDisparity = min_disp,
                                  numDisparities = num_disp,
                                  blockSize = 1,
                                  P1 = 40,
                                  P2 = 100,
                                  disp12MaxDiff = 1,
                                  preFilterCap = 20,
                                  uniquenessRatio = 10,
                                  speckleWindowSize = 100,
                                  speckleRange = 32
    )

    print('computing disparity...')
    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

    print('generating 3d point cloud...',)
    Q = np.array([[1.00000000e+00, 0.00000000e+00, 0.00000000e+00, -9.13191177e+02],
                  [0.00000000e+00, 1.00000000e+00, 0.00000000e+00, -5.39702126e+02],
                  [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 2.98757056e+03],
                  [0.00000000e+00, 0.00000000e+00, 9.94100111e-01, -0.00000000e+00]])
    points = cv.reprojectImageTo3D(disp, Q)
    points[points == (np.inf)] = 0
    points[points == (-np.inf)] = 0
    colors = cv.cvtColor(imgL, cv.COLOR_BGR2RGB)
    mask = disp > disp.min()
    out_points = points[mask]
    out_colors = colors[mask]
    out_fn = 'data/point_cloud/out.ply'
    write_ply(out_fn, out_points, out_colors)
    print('%s saved' % out_fn)


    cv.imshow('left', imgL)
    cv.imshow('disparity', (disp-min_disp)/num_disp)
    cv.waitKey()

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()