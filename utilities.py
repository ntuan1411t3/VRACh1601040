# from cv2 import *
import cv2


def cam_to_img():
    cam = cv2.VideoCapture(0)  # 0 -> index of camera
    s, img = cam.read()
    if s:  # frame captured without any errors
        # cv2.namedWindow("cam-test", cv2.WINDOW_NORMAL)
        # cv2.imshow("cam-test", img)
        # cv2.waitKey(0)
        # cv2.destroyWindow("cam-test")
        # cv2.imwrite("filename.jpg", img)  # save image

        img = cv2.resize(img, (48, 48))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray
        # sift = cv2.xfeatures2d.SIFT_create()
        # kp, des = sift.detectAndCompute(gray, None)
        # cv2.drawKeypoints(gray, kp, outImage=img)
        # cv2.imwrite('filename_gray.jpg', img)


if __name__ == '__main__':
    # initialize the camera
    img = cam_to_img()
    cv2.imwrite('filename_gray.jpg', img)
