import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
def calculate_route(img_name='output.png', boxr=30):
    # Callback function for mouse events
    def mouse_callback(event, x, y, flags, param):
        nonlocal point1, point2, selecting

        if event == cv2.EVENT_LBUTTONDOWN:
            if not selecting:
                point1 = (x, y)
                selecting = True
            else:
                point2 = (x, y)
                selecting = False

                cv2.rectangle(image_copy, point1, point2, (0, 255, 0), 2)
                cv2.imshow('Select Points', image_copy)

    # Read the image
    image = cv2.imread(img_name)
    image_copy = image.copy()

    point1 = (-1, -1)
    point2 = (-1, -1)
    selecting = False

    cv2.namedWindow('Select Points')
    cv2.setMouseCallback('Select Points', mouse_callback)

    while True:
        cv2.imshow('Select Points', image_copy)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # Press 'Esc' key to exit
            break
        elif point1 != (-1, -1) and point2 != (-1, -1):
            break

    cv2.destroyAllWindows()

    if point1 == (-1, -1) or point2 == (-1, -1):
        return None  # Exit if points are not selected

    x0, y0 = point1
    x1, y1 = point2

    rgb_img = plt.imread(img_name)

    if rgb_img.shape.__len__() > 2:
        thr_img = rgb_img[:, :, 0] > np.max(rgb_img[:, :, 0]) / 2
    else:
        thr_img = rgb_img > np.max(rgb_img) / 2

    skeleton = skeletonize(thr_img)
    mapT = ~skeleton

    print("Calculating Route...")
    _mapt = np.copy(mapT)

    if y1 < boxr:
        y1 = boxr
    if x1 < boxr:
        x1 = boxr

    cpys, cpxs = np.where(_mapt[y1 - boxr:y1 + boxr, x1 - boxr:x1 + boxr] == 0)
    cpxs += x1 - boxr
    cpys += y1 - boxr
    idx = np.argmin(np.sqrt((cpys - y1) ** 2 + (cpxs - x1) ** 2))
    y, x = cpys[idx], cpxs[idx]

    pts_x = [x]
    pts_y = [y]
    pts_c = [0]

    xmesh, ymesh = np.meshgrid(np.arange(-1, 2), np.arange(-1, 2))
    ymesh = ymesh.reshape(-1)
    xmesh = xmesh.reshape(-1)

    dst = np.zeros((thr_img.shape))

    while True:
        idc = np.argmin(pts_c)
        ct = pts_c.pop(idc)
        x = pts_x.pop(idc)
        y = pts_y.pop(idc)
        ys, xs = np.where(_mapt[y - 1:y + 2, x - 1:x + 2] == 0)
        _mapt[ys + y - 1, xs + x - 1] = ct
        _mapt[y, x] = 999999
        dst[ys + y - 1, xs + x - 1] = ct + 1
        pts_x.extend(xs + x - 1)
        pts_y.extend(ys + y - 1)
        pts_c.extend([ct + 1] * xs.shape[0])

        if not pts_x:
            break
        if np.sqrt((x - x0) ** 2 + (y - y0) ** 2) < boxr:
            break

    plt.figure(figsize=[14, 14])
    plt.imshow(dst)

    path_x = []
    path_y = []

    while True:
        nbh = dst[y - 1:y + 2, x - 1:x + 2]
        nbh[1, 1] = 9999999
        nbh[nbh == 0] = 9999999

        if np.min(nbh) == 9999999:
            break

        idx = np.argmin(nbh)
        y += ymesh[idx]
        x += xmesh[idx]

        if np.sqrt((x - x1) ** 2 + (y - y1) ** 2) < boxr:
            print("Optimum route found")
            break

        path_y.append(y)
        path_x.append(x)

    plt.figure(figsize=(20, 20))
    plt.imshow(rgb_img)
    plt.plot(path_x, path_y, 'r-', linewidth=5)
    out = "out.jpg"
    plt.savefig(out)
    print("Route saved in: ", out)

def crop_middle(img, crop_size=(1250, 1250)):
    img = cv2.imread(img)
    h, w = img.shape[:2]
    start_row = max(0, (h - crop_size[0]) // 2)
    end_row = start_row + crop_size[0]
    start_col = max(0, (w - crop_size[1]) // 2)
    end_col = start_col + crop_size[1]

    cropped_img = img[start_row:end_row, start_col:end_col]

    return cropped_img
#solve:
def main():
    calculate_route()
    cropped_image = crop_middle("out.jpg")
    cv2.imwrite("out.jpg",cropped_image)
main()