from os import  listdir
from PIL import Image
import numpy as np

#将txt文件转换为向量
def img2vector(filename):
    fobj=open(filename,"r+")
    arr = fobj.readlines()
    vec, demension = [],len(arr)
    for i in range(demension):
        line = arr[i].strip("[]\n ") #去掉[]换行符
        for j in range(demension):
            vec.append((line[j])) #得到向量矩阵
    return vec
 
 #读取训练数据
def createDataset(dir):
    dataset, labels = [], []
    files = listdir(dir) #文件目录列表
    for filename in files:
        label = int(filename[0]) #文件名的第一个字符为数字的标签
        labels.append(label)
        dataset.append(img2vector(dir + '/' + filename)) #读取txt文件并转化为列表元素
    return dataset, labels
 
 #计算谷本系数
def tanimoto(vec1, vec2):
    c1, c2, c3 = 0, 0, 0
    for i in range(len(vec1)):
        if vec1[i] == '0': c1 += 1
        if vec2[i] == '0': c2 += 1
        if vec1[i] == '0' and vec2[i] == "0": c3 += 1
    c1+=1
    return c3 / (c1 + c2 - c3)
    #系数越大，说明相似度越高
#图片转换函数
def imagetotxt():
    imgPath=input("请输入图片：")
    img=Image.open(imgPath).convert("L")
    
    img=img.resize((32,32))
    
    img_new=img.point(lambda x:1 if x>128 else 0)
    img.save("num.png")
    arr=np.array(img_new)
    img_txt=open("./numtxt_test/1.txt","w+")

    for rows in range(arr.shape[0]):
        img_txt.writelines(str(arr[rows].flatten()))
        img_txt.write("\n")
    img_txt.close()



 
#分类器，k值表示选取排序前k组数据
def classify(dataset, labels, testData, k=20): 
    distances = []
    #计算testData与训练样例的相似系数并与相应数字标签存到列表中
    for i in range(len(labels)):
        d = tanimoto(dataset[i], testData)
        distances.append((d, labels[i])) 
    #将相似系数列表按相似系数从大到小排列
    distances.sort(reverse=True)
    klabelDict = {}
    total=0.0
    for i in range(k):
        total+=distances[i][0]
    #对相似系数列表前k项进行概率计算
    check_total=0.0
    for i in range(k):
        klabelDict.setdefault(distances[i][1], 0)
        if check_total<=1.0:
            if distances[i][0]>=0.6:
                klabelDict[distances[i][1]] += 1/2
                check_total+=1/2
            #若字典中有key，返回指定键的值，如果key不在字典中，将会添加键并将值设置为一个指定值0
            else :
                klabelDict[distances[i][1]] += (distances[i][0])/total
                check_total+=(distances[i][0])/total
        
        
 
     #按value降序排序
    predDict = sorted(klabelDict.items(), key=lambda item: item[1], reverse=True)
    return predDict
dataset, labels = createDataset('./numtxt_train')

imagetotxt()
testData = img2vector('./numtxt_test/1.txt')
print("该数字可能数字及对应概率为：\n")
print(classify(dataset, labels, testData))