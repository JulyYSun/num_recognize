from PIL import Image
from os import listdir
import numpy as np
def createTxt(dir):
    
     files = listdir(dir)
     WHITE,BLACK=1,0
     imgtxt=open("./traindata","w+",encoding="utf-8")

     for filename in files:
         num=int(filename[0])
         img=Image.open("./trainimg/{}".format(filename)).convert("L")
         img=img.resize((32,32))
         img_new=img.point(lambda x:BLACK if x>128 else WHITE)
         arr=np.array(img_new)
         arr[0][0]=num  #将每个分块的矩阵的第一个元素设为对应的数字，便于提取标签
         for rows in range(arr.shape[0]):
             imgtxt.writelines(str(arr[rows].flatten()))
             imgtxt.write("\n")
     imgtxt.close()
         
createTxt("./trainimg")
print("转化成功！")
