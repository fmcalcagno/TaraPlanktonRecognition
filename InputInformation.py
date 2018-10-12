import pandas as pd
import UtilsPL

def inputInformation(trainset,testset,transfpos):
    trainlist=pd.read_csv(trainset,delimiter=";",header=1,names =("file","folder","type"))
    testlist=pd.read_csv(testset,delimiter=";",header=1,names =("file","folder","type"))

    hier=pd.DataFrame(UtilsPL.load_obj("hierarchy"),columns=("level1","level2","level3","level4"))
    classes_transf=UtilsPL.load_obj("dictionary_classes")

    hierclasses1=hier['level1'].unique()
    hierclasses2=hier['level2'].unique()
    hierclasses3=hier['level3'].unique()
    hierclasses4=hier['level4'].unique()

    if transfpos==0:
        hierclasses=hierclasses1
    elif transfpos==1:
        hierclasses=hierclasses2
    elif transfpos==2:
        hierclasses=hierclasses3
    elif transfpos==3:
        hierclasses=hierclasses4

    outputsize=len(hierclasses)
    
    return trainlist,testlist,hierclasses,classes_transf,outputsize
