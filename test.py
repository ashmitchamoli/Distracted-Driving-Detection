from lib.driving_dataset.Preprocessor import Preprocessor

preproc = Preprocessor()

data = preproc.readSceneGraphs('test_video')
print(data)