import os
import glob
import shutil
import pandas as pd
import sys

DATASET_DIR = './Datasets/annotatedvideosv1/AnnotatedVideos/'
TARGET_DIR = 'ALL/frames'

runningSceneGraphDfs = []

for videoName in os.listdir(DATASET_DIR):
	if videoName in TARGET_DIR:
		continue

	userId = videoName.split('_')[-2]
	
	# copy all jpeg files to the target directory
	jpgFiles = glob.glob(os.path.join(DATASET_DIR, videoName, '*.jpg'))

	for jpgFile in jpgFiles:
		fileName = jpgFile.split('/')[-1]
		shutil.copy(jpgFile, os.path.join(DATASET_DIR, TARGET_DIR, userId + '_' + fileName))
	
	# read video scene graphs
	sceneGraphs = pd.read_json(os.path.join(DATASET_DIR, videoName, 'sceneGraphs.json'), orient='index')
	runningSceneGraphDfs.append(sceneGraphs)

	# read image embeddings
	imageEmbeddings = pd.read_json(os.path.join(DATASET_DIR, videoName, 'imageEmbeddings.json'), orient='index')
	runningSceneGraphDfs.append(imageEmbeddings)

# concatenate along rows
# sceneGraphs = pd.concat(runningSceneGraphDfs, axis=1)
# sceneGraphs.to_json(os.path.join(DATASET_DIR, TARGET_DIR, 'sceneGraphs.json'), orient='index')

# imageEmbeddings = pd.concat(runningSceneGraphDfs, axis=1)
# imageEmbeddings.to_json(os.path.join(DATASET_DIR, TARGET_DIR, 'imageEmbeddings.json'), orient='index')

print(runningSceneGraphDfs[0])