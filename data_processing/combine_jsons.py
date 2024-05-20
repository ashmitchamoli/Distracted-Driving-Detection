import os
import glob
import shutil
import pandas as pd
import sys

DATASET_DIR = './Datasets/annotatedvideosv1/AnnotatedVideos/'
TARGET_DIR = 'ALL'

runningSceneGraphDfs = []
runningImageEmbeddingDfs = []

for videoName in os.listdir(DATASET_DIR):
	if videoName in TARGET_DIR:
		continue

	# read video scene graphs
	sceneGraphs = pd.read_json(os.path.join(DATASET_DIR, videoName, 'sceneGraphs.json'), orient='index')
	runningSceneGraphDfs.append(sceneGraphs)

	# read image embeddings
	imageEmbeddings = pd.read_json(os.path.join(DATASET_DIR, videoName, 'imageEmbeddings.json'), orient='index')
	runningImageEmbeddingDfs.append(imageEmbeddings)

sceneGraphs = pd.concat(runningSceneGraphDfs, axis=0)
sceneGraphs.to_json(os.path.join(DATASET_DIR, TARGET_DIR, 'sceneGraphs.json'), orient='index')

imageEmbeddings : pd.DataFrame = pd.concat(runningImageEmbeddingDfs, axis=0)
imageEmbeddings.to_json(os.path.join(DATASET_DIR, TARGET_DIR, 'imageEmbeddings.json'), orient='index')
