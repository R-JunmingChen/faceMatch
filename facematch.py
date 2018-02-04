# -*- coding: utf-8 -*-
import dlib
import numpy as np
import cv2
import dlib
import glob
import pickle
import os
import lshash

prepath = "./model/"
detector = dlib.get_frontal_face_detector()
predictor_path = prepath+'shape_predictor_landmarks.dat'
face_rec_model_path =prepath+ 'face_recognition_model.dat'
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

def __getsingledescriptor_fromimg(imgpath):
    """
    :parameter imgpath  single img path
    :return the first face descriptor calculated by dlib,which is a python bultd-in list
    """
    image = cv2.imread(imgpath)
    dets = detector(image, 1)
    testdescriptors = []
    for index, item in enumerate(dets):
        shape = sp(image, item)
        face_descriptor = facerec.compute_face_descriptor(image, shape)
        testdescriptors.append(face_descriptor)
        break

    return testdescriptors[0]


def __getdescriptors_fromdir(dirpath):
    """
    :param dirpath: the path of img dir
    :return: a dic,whic key is img file name and value is descriptor
    """
    dic={}
    for filepath in glob.glob(os.path.join(dirpath, "*.jpg")):
        filename = filepath.split("/")[-1].split(".")[0]
        if "\\" in filename:
            filename = filename.split("\\")[-1]

        descriptor=__getsingledescriptor_fromimg(filepath)
        dic[filename]=descriptor

    return dic




def buildmodel(imgdir,modeldir):
    """
    the function would read imgs, and generate (1)descriptors of img, (2)name of img,
    (3)lsh of descriptors , (4)reverse index of lsh index. Finally, the models would be saved in modeldir
    :param imgdir:the dir saving img
    :param modeldir:the dir saving model
    :return:not return
    """
    descriptors_dic=__getdescriptors_fromdir(imgdir)
    descriptors=[]
    names=[]
    for k,v in descriptors_dic.items():
        names.append(k)
        descriptors.append(v)

    filedescriptors = open(os.path.join(modeldir, "descriptors.pkl"),'wb')
    pickle.dump(descriptors, filedescriptors)
    filenames = open(os.path.join(modeldir, "names.pkl"),'wb')
    pickle.dump(names, filenames)


    # descriptors lsh
    lsh = lshash.LSHash(30, 128, num_hashtables=2)
    dic_Reverse = {}
    for i in range(0, len(descriptors)):
        lsh.index(descriptors[i])
        key = str(descriptors[i][0] + descriptors[i][1]) + str(descriptors[i][2] + descriptors[i][3])
        dic_Reverse[key] = i

    filelsh = open(os.path.join(modeldir, "lsh.pkl"), 'wb')
    pickle.dump(lsh, filelsh)
    filedic_Reverse = open(os.path.join(modeldir, "dic_Reverse.pkl"), 'wb')
    pickle.dump(dic_Reverse, filedic_Reverse)





def facematch(imgpath,modeldir):
    """
    the match is based on Euclidean distance
    :param imgpath: imgpath which needs to match
    :param modeldir: the dir saving the models
    :return:match_pic_name, match_similarity
    """
    #load model
    descriptors=pickle.load(open( os.path.join(modeldir, "descriptors.pkl"),'rb'))
    names = pickle.load(open(os.path.join(modeldir, "names.pkl"), 'rb'))
    lsh = pickle.load(open(os.path.join(modeldir, "lsh.pkl"), 'rb'))
    dic_Reverse = pickle.load(open(os.path.join(modeldir, "dic_Reverse.pkl"), 'rb'))

    #test img
    testdescriptorVector = __getsingledescriptor_fromimg(imgpath)
    testdescriptor = np.array(testdescriptorVector)

    # use lsh to find
    while 1:
        # lsh 匹配到相应descriptor
        matchdescriptorresultes = lsh.query(testdescriptorVector)
        if len(matchdescriptorresultes) <= 0:
            break
        #choose the the first result(the nearst distance result),and choose the first element of result(descriptor),the second element is distance
        matchresult=matchdescriptorresultes[0][0]
        matchdescriptor = np.array(matchresult)

        eudis = __eudistance(testdescriptor, matchdescriptor)
        sim = __cossim(testdescriptor, matchdescriptor)
        if eudis < 0.55:
            return  names[dic_Reverse[__getkey(matchresult)]], sim
        else:
            break

    # lsh failed, iter all the descriptors to compare
    mineudis = 9999
    minindex = -1
    for index, descriptorVector in enumerate(descriptors):
        matchdescriptor = np.array(descriptorVector)
        eudis = __eudistance(testdescriptor, matchdescriptor)

        if eudis < mineudis:
            mineudis = eudis
            minindex = index

    sim = __cossim(testdescriptor, np.array(descriptors[minindex]))
    name = names[minindex]

    return  name, sim


def __getkey(vector):
    return str(vector[0] + vector[1]) + str(vector[2] + vector[3])


def __cossim(vectorA, vectorB):
    vectorA = vectorA.reshape([128])
    vectorB = vectorB.reshape([128])
    a_model = np.sqrt(np.sum(np.square(vectorA)))
    b_model = np.sqrt(np.sum(np.square(vectorB)))
    tem = np.dot(vectorA, vectorB)
    return tem / (a_model * b_model)


def __eudistance(vectorA, vectorB):
    return np.sqrt(np.sum(np.square(vectorA - vectorB)))


if __name__=="__main__":
    #这是一个建立匹配库的实例
    buildmodel("./data", "./model")

    #这是一个匹配的实例
    name, sim=facematch("./data/4.jpg","./model")
    print(str("name:"+str( name)+" sim: "+str( sim)))