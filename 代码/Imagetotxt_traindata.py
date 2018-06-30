from PIL import Image
from os import listdir
import numpy as np
def createTxt(dir):
    
     files = listdir(dir)
     WHITE,BLACK=1,0
     for filename in files:
         img=Image.open("./num_train/{}".format(filename)).convert("L")
         img=img.resize((32,32))
         img_new=img.point(lambda x:WHITE if x>128 else BLACK)
         arr=np.array(img_new)
         newfilename=filename[:4]
         imgtxt=open("./numtxt_train/{}.txt".format(newfilename),"w+",encoding="utf-8")
         for rows in range(arr.shape[0]):
             imgtxt.writelines(str(arr[rows].flatten()))
             imgtxt.write("\n")
         imgtxt.close()
         
createTxt("./num_train")
print("转化成功！")
