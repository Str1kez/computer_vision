import cv2
import numpy as np
from matplotlib import pyplot as plt


def filter_sobel(image):
    H, W = image.shape

    out = np.zeros((H + 2, W + 2, 2), np.float64)
    out[1: 1 + H, 1: 1 + W, 0] = image.copy().astype(np.float64)
    out[0, 0, 0] = out[1, 1, 0]
    out[-1, 0, 0] = out[-2, 1, 0]
    out[0, -1, 0] = out[1, -2, 0]
    out[-1, -1, 0] = out[-2, -2, 0]
    out[1: 1 + H, 0, 0] = out[1: 1 + H, 1, 0]
    out[1: 1 + H, -1, 0] = out[1: 1 + H, -2, 0]
    out[0, 1: 1 + W, 0] = out[1, 1: 1 + W, 0]
    out[-1, 1: 1 + W, 0] = out[-2, 1: 1 + W, 0]

    Mx = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]],
        dtype=int
    )
    My = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ],
        dtype=int
    )
    tmp = out.copy()

    # Ix, Iy
    for y in range(H):
        for x in range(W):
            out[1 + y, 1 + x] = (np.sum(Mx * tmp[y: y + 3, x: x + 3, 0]), np.sum(My * tmp[y: y + 3, x: x + 3, 0]))
    return out[1: 1 + H, 1: 1 + W]


def my_harris(img_dir, window_size, k, threshold):
    img = cv2.imread(img_dir)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_gaussian = cv2.GaussianBlur(gray, (3, 3), 0)

    # Check if the image is exists
    if img is None:
        print('Invalid image:' + img_dir)
        return None
    else:
        print('Image successfully read...')

    height = img.shape[0]  # .shape[0] outputs height
    width = img.shape[1]  # .shape[1] outputs width .shape[2] outputs color channels of image
    matrix_R = np.zeros((height, width))

    #   Step 1 - Calculate the x e y image derivatives (dx e dy)
    # dx = cv2.Sobel(img_gaussian, cv2.CV_64F, 1, 0, ksize=3)
    # dy = cv2.Sobel(img_gaussian, cv2.CV_64F, 0, 1, ksize=3)
    gradients = filter_sobel(gray)
    dx, dy = gradients[:, :, 0], gradients[:, :, 1]
    # dy, dx = np.gradient(gray)

    #   Step 2 - Calculate product and second derivatives (dx2, dy2 e dxy)
    # dx2 = np.square(dx)
    # dy2 = np.square(dy)
    dxy = dx * dy

    offset = int(window_size / 2)
    #   Step 3 - Calcular a soma dos produtos das derivadas para cada pixel (Sx2, Sy2 e Sxy)
    print("Finding Corners...")
    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            Sx2 = np.sum(dx[y - offset:y + 1 + offset, x - offset:x + 1 + offset] ** 2)
            Sy2 = np.sum(dy[y - offset:y + 1 + offset, x - offset:x + 1 + offset] ** 2)
            Sxy = np.sum(dxy[y - offset:y + 1 + offset, x - offset:x + 1 + offset])

            #   Step 4 - Define the matrix H(x,y)=[[Sx2,Sxy],[Sxy,Sy2]]
            H = np.array([[Sx2, Sxy], [Sxy, Sy2]])

            #   Step 5 - Calculate the response function ( R=det(H)-k(Trace(H))^2 )
            R = np.linalg.det(H) - k * np.matrix.trace(H) ** 2
            matrix_R[y - offset, x - offset] = R

    #   Step 6 - Apply a threshold
    # cv2.normalize(matrix_R, matrix_R, 0, 1, cv2.NORM_MINMAX)
    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            value = matrix_R[y, x]
            if value > threshold:
                # cornerList.append([x, y, value])
                cv2.circle(img, (x, y), 3, (0, 255, 0))

    # cv2.imwrite("%s_threshold_%s.png"%(img_dir[5:-4],threshold), img)
    plt.figure("Manually implemented Harris detector")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title("Manually implemented Harris detector")
    plt.xticks([]), plt.yticks([])
    # plt.savefig('My_harris_detector-thresh_%s.png' % threshold, bbox_inches='tight')
    plt.show()


my_harris("./images/phpDWi1Qn.png", 5, 0.04, 0.30)  # Change this path to one that will lead to your image
