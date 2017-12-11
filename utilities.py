from cv2 import *

if __name__ == '__main__':

    # initialize the camera
    cam = VideoCapture(0)  # 0 -> index of camera
    s, img = cam.read()
    if s:  # frame captured without any errors
        namedWindow("cam-test", WINDOW_NORMAL)
        imshow("cam-test", img)
        waitKey(0)
        destroyWindow("cam-test")
        imwrite("filename.jpg", img)  # save image

        img = resize(img, (48, 48))
        gray = cvtColor(img, COLOR_BGR2GRAY)
        sift = xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(gray, None)
        drawKeypoints(gray, kp, outImage=img)
        imwrite('filename_gray.jpg', img)
