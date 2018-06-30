from PIL import Image
from matplotlib import pyplot
import numpy as np
img=Image.open("E:\\学习\\离散数学\\小组展示\\中国地图.png")
#img=img.resize()
pyplot.imshow(img)
pyplot.show()