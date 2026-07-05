import cv2
import matplotlib.pyplot as plt
import random

DATA_PATH = r"C:\university\SHANA 5\semester B\67604-slam\VAN_SLAM\VAN_ex\dataset\dataset_2026\sequences\00"
def read_images(idx):
    img_name = '{:06d}.png'.format(idx)
    img1 = cv2.imread(DATA_PATH+'\\image_0\\'+img_name, 0)
    img2 = cv2.imread(DATA_PATH+'\\image_1\\'+img_name, 0)
    return img1, img2

if __name__ == '__main__':
    ##### 1.1 ######
    img1, img2 = read_images(000000)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    img1_with_kp = cv2.drawKeypoints(img1, kp1, None, color=(0, 255, 255))
    img2_with_kp = cv2.drawKeypoints(img2, kp2, None, color=(0, 255, 255))

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(img1_with_kp)
    plt.title(f'image_0: {len(kp1)} points')

    plt.subplot(1, 2, 2)
    plt.imshow(img2_with_kp)
    plt.title(f'image_1: {len(kp2)} points')
    plt.show()

    ##### 1.2 #####
    print("descriptor of the first feature")
    print(des1[0])
    print("descriptor of the second feature")
    print(des1[1])

    ##### 1.3 #####
    bf = cv2.BFMatcher()
    matches = bf.match(des1, des2)
    random_matches = random.sample(matches, 20)

    img3_with_bf = cv2.drawMatches(img1, kp1, img2, kp2,random_matches,None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3_with_bf)
    plt.title("20 random matches #1.3#")
    plt.show()

    ##### 1.4 #####
    matches_new = bf.knnMatch(des1, des2, k=2)
    ratio = 0.7
    good_matches = []
    discarded_matches = []
    discarded_count = 0

    for m, n in matches_new:
        if m.distance < ratio * n.distance:
            good_matches.append(m)
        else:
            discarded_count += 1
            discarded_matches.append(m)

    print("the ratio we use: ", ratio)
    print("we discarded {:d} matches".format(discarded_count))

    #the good match
    random_good_matches = random.sample(good_matches, 20)
    img_good_result = cv2.drawMatches(img1, kp1, img2, kp2, random_good_matches, None, flags=2)
    plt.imshow(img_good_result)
    plt.title("A 20 correct matchs #1.4#")
    plt.show()

    #the discarded match
    index_to_test = 50 #I choose randomly until I get good match
    failed_match = discarded_matches[index_to_test]
    img_failed_match = cv2.drawMatches(img1, kp1,img2, kp2, [failed_match],
                                       None,flags=2, matchColor=(0, 255, 0))
    plt.imshow(img_failed_match)
    plt.title("A Correct Match that Failed the Ratio Test #1.4#")
    plt.show()

