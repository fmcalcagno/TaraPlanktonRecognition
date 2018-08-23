import pandas as pd
import UtilsPL

def inputInformation(trainset,testset,transfpos):
    trainlist=pd.read_csv(trainset,delimiter=";",header=1,names =("file","folder","type"))
    testlist=pd.read_csv(testset,delimiter=";",header=1,names =("file","folder","type"))

    hier=pd.DataFrame(UtilsPL.load_obj("hierarchy"),columns=("living","parent","child"))
    classes_transf=UtilsPL.load_obj("dictionary_classes")

    hierclasses1=hier['living'].unique()
    hierclasses2=hier['parent'].unique()
    hierclasses3=hier['child'].unique()

    if transfpos==0:
        hierclasses=hierclasses1
    elif transfpos==1:
        hierclasses=hierclasses2
    elif transfpos==2:
        hierclasses=hierclasses3

    outputsize=len(hierclasses)
    
    return trainlist,testlist,hierclasses,classes_transf,outputsize
