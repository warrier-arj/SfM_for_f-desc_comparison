"""
Arjun Rajeev Warrier and Sahil Kiran Bodke
Feature detection happens here for comparison
Final Project PRCV
match program: run this to execute feature descriptor comparison
"""
import cv2 as cv
import matplotlib.pyplot as plt
import time

class Match:
    def __init__(self, img_path1, img_path2, kp_detector, kp, desc):
        self.img1 = cv.imread(img_path1, cv.IMREAD_GRAYSCALE) # queryImage
        self.img2 = cv.imread(img_path2, cv.IMREAD_GRAYSCALE) # trainImage
        self.kp_detector = kp_detector
        self.kp = kp
        self.desc = desc
        self.matches = None

        self.find_matches()
        self.plot_matches()
        #self.save_matches()

    def find_matches(self, ratio_threshold=2):
        start_time = time.time()

        # BFMatcher with default params
        bf = cv.BFMatcher()
        matches = bf.knnMatch(self.desc, self.desc, k=2)

        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < ratio_threshold * n.distance:
                good_matches.append([m])

        self.matches = good_matches

        # Print time taken and number of matches
        end_time = time.time()
        print("Time taken for matching " + self.kp_detector + " features: ", end_time - start_time)
        print("Number of matches " + self.kp_detector + " features: ", str(self.num_matches()))

    def draw_matches(self, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS):
        img3 = cv.drawMatchesKnn(self.img1, self.kp, self.img2, self.kp, self.matches, None, flags=flags)

        return img3

    def save_matches(self, img_path):
        cv.imwrite(img_path, self.draw_matches())

    def plot_matches(self):
        plt.imshow(self.draw_matches())
        plt.title(self.kp_detector + " feature matches")
        plt.show()


    def num_matches(self):
        return len(self.matches)
