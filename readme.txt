-----------------------------------------------------------------------------------------------------------------
Topic: Implementation of SfM and comparison of feature descriptors
-----------------------------------------------------------------------------------------------------------------

Project Description:
Our project aims to develop a Python-based structure from motion pipeline, incorporating various feature descriptors for comparison purposes. To achieve this, we will be referencing OpenSfm and COLMAP documentations, as well as other relevant publications to determine appropriate comparison metrics for the feature descriptors. Additionally, we plan to explore different types of RANSAC algorithms to augment our pipeline, subject to time constraints.
-----------------------------------------------------------------------------------------------------------------


Code Particulars:

Two Main strucures have been included:
1) To compare feature descriptors: Run draw_features.py with match_features.py in the same folder
					     Change filepaths in draw_features.py before running to match different images
2) For SfM: run main.py with reconstruction.py, matching.py and bundle_adjustment.py.
		To run a different dataaset include in if condition in matching and change name in main.py

File Directory:
2) Dataset(not included): 
- Middlebury dataset has been used.
- The dataset follows a specific directory structure which is as follows:
 
- There is a folder 'datasets' inside which there are individual folders for different image datasets such as   'templeRing' and 'Viking' which are named according to the actual dataset that they belong to. 

- Finally, all the images are stored within these individual folders. The images follow a 'zero-padded' naming convention which means that the first image is named as '00.jpg', the second image is named as '01.jpg', and so on. The file extensions can either be '.jpg' or '.png' and all the images belonging to a particular dataset must be of the same size. 

- Store scripts immediately outside the datasets folder. 


-----------------------------------------------------------------------------------------------------------------
