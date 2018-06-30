from PIL import Image
from os import listdir
import numpy as np


#将txt文件转换为向量

def getTxt(traindata):
    fobj=open(traindata,"r+")
    arr_total=fobj.readlines() #读取traindata中所有行的字符
    arrlen=len(arr_total) 
    
    dataset=[]
    label=[]
    dimension=32
    arrcount=arrlen/dimension #训练列表中元素个数，也就是用于训练的图片个数
    for i in range(int(arrcount)):
        numindex=arr_total[i*dimension][1] #从矩阵第一个元素获取数字的标签
        label.append(numindex) #将数字标签添加到列表中
        vec=[] #存储每32行的字符

        #每32行获取一个数字矩阵
        for j in range(i*dimension,(i+1)*dimension):
            line=arr_total[j].strip("[]\n ")
            for k in range(dimension):
                vec.append((line[k]))
        dataset.append(vec) #将每一个数字矩阵化为一维矩阵添加到数据集列表中
    
    return dataset,label #返回数据集，标签列表

def img2vector(filename):
    fobj=open(filename,"r+")
    arr = fobj.readlines()
    vec, dimension = [],len(arr)
    for i in range(dimension):
        line = arr[i].strip("[]\n ") #去掉[]换行符
        for j in range(dimension):
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
    
    if (c1+c2-c3)==0:
        return 1
    else :
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
    img_txt=open("./numtxt_test/testnumber","w+")

    for rows in range(arr.shape[0]):
        img_txt.writelines(str(arr[rows].flatten()))
        img_txt.write("\n")
    img_txt.close()

#分类器，k值表示选取排序前k组数据
def classify(dataset, labels, testData, k=50): 
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
            if distances[i][0]>=0.75:
                klabelDict[distances[i][1]] += 1/2
                check_total+=1/2
            #若字典中有key，返回指定键的值，如果key不在字典中，将会添加键并将值设置为一个指定值0
            else :
                klabelDict[distances[i][1]] += (distances[i][0])/total
                check_total+=(distances[i][0])/total
        
        
 
     #按value降序排序
    predDict = sorted(klabelDict.items(), key=lambda item: item[1], reverse=True)
    return predDict[0][0]
def createTxt(dir):
    dataset, labels = getTxt('traindata')
    files = listdir(dir)
    img_total=len(files)
    count=0
    for filename in files:
        img=Image.open('./num_train/{}'.format(filename)).convert("L")
        num_real=int(filename[0])
        img=img.resize((32,32))
        
        img_new=img.point(lambda x:1 if x>128 else 0)
        
        arr=np.array(img_new)
        img_txt=open("./numtxt_test/testnumber","w+")

        for rows in range(arr.shape[0]):
            img_txt.writelines(str(arr[rows].flatten()))
            img_txt.write("\n")
        img_txt.close()
        testData = img2vector('./numtxt_test/testnumber')
        num_predict=int(classify(dataset,labels,testData))
        if num_predict==num_real:
            count+=1
    print("正确率为：")
    print(count/img_total)

createTxt('num_train')



