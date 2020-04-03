import cv2
import numpy as np
from pathlib import Path

def cvtGray(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img

def findKeypoints(sift,img):
    kp,dst = cv2.xfeatures2d_SIFT.detectAndCompute(sift,img,None)
    return kp,dst

def findFeatures(feature):
    cv2.FlannBasedMatcher.knnMatch(feature,)

def findMatches(matches):
    good = []
    for (m, n) in matches:
        if m.distance < n.distance * 0.7:
            good.append(m)
    return good

if __name__ == "__main__":
    query_img = cv2.imread("/Users/prituldave/Downloads/code/book_covers/queries/query06.png")
    pathlist = Path("/Users/prituldave/Downloads/code/book_covers/covers").glob('*.png')
    query_img_gray = cvtGray(query_img)

    cv2.imshow("query image",query_img)
    cv2.waitKey(0)
    sift = cv2.xfeatures2d_SIFT.create()
    kp1,dst1 = findKeypoints(sift,query_img)

    feature = cv2.FlannBasedMatcher_create()
    flag = 0
    print("processing\n.....................")
    for path in pathlist:
        testing_img = cv2.imread(str(path))
        #testing_img = cv2.imread("/Users/prituldave/Downloads/code/book_covers/covers/cover016.png")
        testing_gray = cvtGray(testing_img)

        kp2,dst2 = findKeypoints(sift,testing_gray)

        matches = cv2.FlannBasedMatcher.knnMatch(feature,dst1,dst2,2)

        good = findMatches(matches)



        if len(good) > 130:
            print("completed")
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()

            #resizing the testing image#
            if testing_img.shape[0]>400 or testing_img.shape[1]>400:
                testing_img = cv2.resize(testing_img,(400,400))

            #cv2.imshow("resultant image", testing_img)

            #placing the name on the image#
            file_name = str(path.name)[:-4]

            cv2.putText(testing_img,file_name,(20,30),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

            cv2.imshow("query image", query_img)
            cv2.imshow("resultant image",testing_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            flag = 1
            break
        else:

            flag=0
            matchesMask = None
    #The book is not found#
    if flag!=1:
        print("completed")
        cv2.putText(query_img,"unable to find book" , (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("query image", query_img)
        cv2.waitKey(0)
cv2.destroyAllWindows()