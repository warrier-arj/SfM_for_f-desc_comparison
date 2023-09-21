"""
Arjun Rajeev Warrier and Sahil Kiran Bodke
Feature  matching happens here for feature descriptor comparison
Final Project PRCV
match program: run draw_features.py to execute sfm
"""
import cv2
import time
from draw_features import *


class ft_desc:
    """
    This function uses the SIFT algorithm to detect and compute keypoints and descriptors for the grayscale image.
    """
    def __init__(self, img_path, ctr):
        self.img = cv2.imread(img_path)
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.kp_sift = None
        self.kp_brisk = None
        self.kp_orb = None
        self.desc_sift = None
        self.desc_brisk = None
        self.desc_orb = None
        self.sift()
        self.brisk()
        self.orb()
        if ctr:
            self.draw_kp()

    def sift(self):
        sift = cv2.SIFT_create()

        # Compute SIFT descriptor and measure time
        start_time = time.time()
        self.kp_sift, self.desc_sift = sift.detectAndCompute(self.gray, None)
        end_time = time.time()

        # Print time taken and number of features
        print("SIFT time:", end_time - start_time)
        print("SIFT key-points:", len(self.kp_sift))

    def brisk(self):
        """
        This function uses the BRISK algorithm to detect and compute keypoints and descriptors for the grayscale image.
        """
        brisk = cv2.BRISK_create()

        start_time = time.time()
        self.kp_brisk, self.desc_brisk = brisk.detectAndCompute(self.gray, None)
        end_time = time.time()

        # Print time taken
        print("BRISK time:", end_time - start_time)
        print("BRISK key-points:", len(self.kp_brisk))

    def orb(self):
        """
         This function uses the ORB algorithm to detect and compute keypoints and descriptors for the grayscale image.
        """
        orb = cv2.ORB_create(nfeatures = 1000)

        start_time = time.time()
        self.kp_orb, self.desc_orb = orb.detectAndCompute(self.gray, None)
        end_time = time.time()

        # Print time taken
        print("ORB time:", end_time - start_time)
        print("ORB key-points:", len(self.kp_orb))

    def draw_kp(self):
        """
        This function creates a subplot with the original image and the keypoints for each algorithm
        and shows it using OpenCV.
        """
        img_kp_sift = cv2.drawKeypoints(self.gray, self.kp_sift, None, color=(0, 255, 0))
        img_kp_brisk = cv2.drawKeypoints(self.gray, self.kp_brisk, None, color=(0, 0, 255))
        img_kp_orb = cv2.drawKeypoints(self.gray, self.kp_orb, None, color=(255, 0, 0))

        fig, ax = plt.subplots(2, 2, figsize=(10, 8))
        ax[0, 0].imshow(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
        ax[0, 0].set_title("Original Image")
        ax[0, 1].imshow(img_kp_sift)
        ax[0, 1].set_title("SIFT Keypoints")
        ax[1, 0].imshow(img_kp_brisk)
        ax[1, 0].set_title("BRISK Keypoints")
        ax[1, 1].imshow(img_kp_orb)
        ax[1, 1].set_title("ORB Keypoints")

        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.show()

# Main function
def main():
    # Image paths
    img_path1 = "../datasets/templeRing/03.png"
    img_path2 = "../datasets/templeRing/08.png"

    # Feature detection
    fd = ft_desc(img_path1, 1)
    fd2 = ft_desc(img_path2, 0)

    # Keypoint matching
    match_sift = Match(img_path1, img_path2, 'sift', fd.kp_sift, fd.desc_sift)
    match_brisk = Match(img_path1, img_path2, 'brisk', fd.kp_brisk, fd.desc_brisk)
    match_orb = Match(img_path1, img_path2, 'orb', fd.kp_orb, fd.desc_orb)

if __name__ == '__main__':
    main()
