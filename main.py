"""
Arjun Rajeev Warrier and Sahil Kiran Bodke
Incremental SfM
Final Project PRCV
main program: run to execute sfm
"""
import cv2
import pyntcloud
import random
import numpy as np
import open3d as o3d
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import bundle_adjustment as b
import matching as m
import reconstruction as r
import time

mpl.rcParams['figure.dpi']= 200

def main():
    start_time = time.time()
    # Matching between images and outlier removal
    n_imgs = 46 # 46 if imgset = 'templering', 49 if imgset = 'Viking'
    imgset = 'templering'

    images, keypoints, descriptors, K = m.find_features(n_imgs, imgset)

    # Brute force matcher
    matcher = cv2.BFMatcher(cv2.NORM_L1)

    # Flann based matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)


    matches = m.find_matches(matcher, keypoints, descriptors)
    print('num_matches before outlier removal:', m.num_matches(matches))
    m.print_num_img_pairs(matches)

    matches = m.remove_outliers(matches, keypoints)
    print("After outlier removal:")
    m.print_num_img_pairs(matches)

    img_adjacency, list_of_img_pairs = m.create_img_adjacency_matrix(n_imgs, matches)

    ### This cell initializes the reconstruction
    best_pair = r.best_img_pair(img_adjacency, matches, keypoints, K, top_x_perc=0.2)
    R0, t0, R1, t1, points3d_with_views = r.initialize_reconstruction(keypoints, matches, K, best_pair[0], best_pair[1])

    R_mats = {best_pair[0]: R0, best_pair[1]: R1}
    t_vecs = {best_pair[0]: t0, best_pair[1]: t1}

    resected_imgs = [best_pair[0], best_pair[1]]
    unresected_imgs = [i for i in range(len(images)) if i not in resected_imgs]
    print('initial image pair:', resected_imgs)
    avg_err = 0

    ### This cell grows and refines the reconstruction
    BA_chkpts = [3, 4, 5, 6] + [int(6 * (1.34 ** i)) for i in range(25)]
    while len(unresected_imgs) > 0:
        resected_idx, unresected_idx, prepend = r.next_img_pair_to_grow_reconstruction(n_imgs, best_pair, resected_imgs,
                                                                                       unresected_imgs, img_adjacency)
        points3d_with_views, pts3d_for_pnp, pts2d_for_pnp, triangulation_status = r.get_correspondences_for_pnp(
            resected_idx, unresected_idx, points3d_with_views, matches, keypoints)
        if len(pts3d_for_pnp) < 12:
            print(
                f"{len(pts3d_for_pnp)} is too few correspondences for pnp. Skipping imgs resected:{resected_idx} and unresected:{unresected_idx}")
            print(f"Currently resected imgs: {resected_imgs}, unresected: {unresected_imgs}")
            continue

        R_res = R_mats[resected_idx]
        t_res = t_vecs[resected_idx]
        print(f"Unresected image: {unresected_idx}, resected: {resected_idx}")
        R_new, t_new = r.do_pnp(pts3d_for_pnp, pts2d_for_pnp, K)
        R_mats[unresected_idx] = R_new
        t_vecs[unresected_idx] = t_new
        if prepend == True:
            resected_imgs.insert(0, unresected_idx)
        else:
            resected_imgs.append(unresected_idx)
        unresected_imgs.remove(unresected_idx)
        pnp_errors, projpts, avg_err, perc_inliers = r.test_reproj_pnp_points(pts3d_for_pnp, pts2d_for_pnp, R_new, t_new, K)
        print(f"Average error of reprojecting points used to resect image {unresected_idx} back onto it is: {avg_err}")
        print(f"Fraction of Pnp inliers: {perc_inliers} num pts used in Pnp: {len(pnp_errors)}")

        if resected_idx < unresected_idx:
            kpts1, kpts2, kpts1_idxs, kpts2_idxs = r.get_aligned_kpts(resected_idx, unresected_idx, keypoints, matches,
                                                                      mask=triangulation_status)
            if np.sum(triangulation_status) > 0:  # at least 1 point needs to be triangulated
                points3d_with_views, tri_errors, avg_tri_err_l, avg_tri_err_r = r.triangulate_points_and_reproject(R_res,
                                                                                                                   t_res,
                                                                                                                   R_new,
                                                                                                                   t_new, K,
                                                                                                                   points3d_with_views,
                                                                                                                   resected_idx,
                                                                                                                   unresected_idx,
                                                                                                                   kpts1,
                                                                                                                   kpts2,
                                                                                                                   kpts1_idxs,
                                                                                                                   kpts2_idxs,
                                                                                                                   reproject=True)
        else:
            kpts1, kpts2, kpts1_idxs, kpts2_idxs = r.get_aligned_kpts(unresected_idx, resected_idx, keypoints, matches,
                                                                      mask=triangulation_status)
            if np.sum(triangulation_status) > 0:  # at least 1 point needs to be triangulated
                points3d_with_views, tri_errors, avg_tri_err_l, avg_tri_err_r = r.triangulate_points_and_reproject(R_new,
                                                                                                                   t_new,
                                                                                                                   R_res,
                                                                                                                   t_res, K,
                                                                                                                   points3d_with_views,
                                                                                                                   unresected_idx,
                                                                                                                   resected_idx,
                                                                                                                   kpts1,
                                                                                                                   kpts2,
                                                                                                                   kpts1_idxs,
                                                                                                                   kpts2_idxs,
                                                                                                                   reproject=True)

        if 0.8 < perc_inliers < 0.95 or 5 < avg_tri_err_l < 10 or 5 < avg_tri_err_r < 10:
            # If % of inlers from Pnp is too low or triangulation error on either image is too high, bundle adjust
            points3d_with_views, R_mats, t_vecs = b.do_BA(points3d_with_views, R_mats, t_vecs, resected_imgs, keypoints, K,
                                                          ftol=1e0)

        if len(resected_imgs) in BA_chkpts or len(
                unresected_imgs) == 0 or perc_inliers <= 0.8 or avg_tri_err_l >= 10 or avg_tri_err_r >= 10:
            # If % of inlers from Pnp is very low or triangulation error on either image is very high, bundle adjust with stricter tolerance
            points3d_with_views, R_mats, t_vecs = b.do_BA(points3d_with_views, R_mats, t_vecs, resected_imgs, keypoints, K,
                                                          ftol=1e-1)

        av = 0
        for im in resected_imgs:
            p3d, p2d, avg_error, errors = r.get_reproj_errors(im, points3d_with_views, R_mats[im], t_vecs[im], K, keypoints,
                                                              distCoeffs=np.array([]))
            print(f'Average reprojection error on image {im} is {avg_error} pixels')
            av += avg_error
        av = av / len(resected_imgs)
        print(f'Average reprojection error across all {len(resected_imgs)} resected images is {av} pixels')

    ### This cell visualizes the pointcloud
    num_voxels = 200  # Set to 100 for faster visualization, 200 for higher resolution.
    x, y, z = [], [], []
    for pt3 in points3d_with_views:
        if abs(pt3.point3d[0][0]) + abs(pt3.point3d[0][1]) + abs(pt3.point3d[0][2]) < 100:
            x.append(pt3.point3d[0][0])
            y.append(pt3.point3d[0][1])
            z.append(pt3.point3d[0][2])
    vpoints = np.vstack((x, y, z)).T

    # Create a PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vpoints)

    # Save the PointCloud to a PLY file
    o3d.io.write_point_cloud("point_cloud.ply", pcd)

    # Overall time taken
    end_time = time.time()
    print("Time taken for " + imgset + ":" + str(end_time-start_time))



if __name__ == '__main__':
    main()