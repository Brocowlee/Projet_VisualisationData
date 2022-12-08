import pandas as pd
import plotly.express as px
import json
from dash import Dash, dcc, html, Input, Output
from plotly.subplots import make_subplots
import os

files_name = ['D:\Documents\WORK\IDU4\VISUALISATION//' + file_name for file_name in os.listdir('D:\Documents\WORK\IDU4\VISUALISATION') if '.json' in file_name]


def KW100(link1,link2):
    files_name=[link1,link2]
    allKw=[]
    for file_name in files_name:
        keywords_count = []
        print(file_name)
        f = open(file_name, 'r', encoding='utf-8')
        data = json.loads(f.read())
        for fr in data['metadata-all'].values():
            for year in fr["year"].items():
                for keyword in year[1]["kws"].items():
                    keywords_count.append( [str(keyword[0]), int(keyword[1]) ])
                
        df = pd.DataFrame.from_dict(keywords_count).rename(columns = {0:"kw",1:"amount"})
        df = df.groupby(["kw"]).mean()
        df= df.nlargest(200, 'amount')
        #df = df.reset_index()
        allKw.append(df)

    
    return allKw[0],allKw[1]


tab1=KW100("D:/Documents/WORK/IDU4/VISUALISATION/topaz-data732--mali--www.fdesouche.com--20190101--20211231.json","D:/Documents/WORK/IDU4/VISUALISATION/topaz-data732--mali--www.egaliteetreconciliation.fr--20190101--20211231.json")[0]
tab2=KW100("D:/Documents/WORK/IDU4/VISUALISATION/topaz-data732--mali--www.fdesouche.com--20190101--20211231.json","D:/Documents/WORK/IDU4/VISUALISATION/topaz-data732--mali--www.egaliteetreconciliation.fr--20190101--20211231.json")[1]

# print(tab1)
# print(tab2)

def combine(kw1,kw2):
    res=pd.concat([kw1,kw2])
    res=pd.DataFrame(index=res.index.drop_duplicates(),columns=["amount1","amount2"]).fillna(0)
    print(res)
    for e in res.index:
        try:
            res.at[e,"amount1"]=kw1.at[e,"amount"]
        except:
            pass
        try:
            res.at[e,"amount2"]=kw2.at[e,"amount"]
        except:
            pass
    return res

print(combine(tab1,tab2))



def interetCommun(kw1,kw2):
    df=combine(tab1,tab2)
    df["exclu"]=abs(df["amount1"]-df["amount2"])
    fig=px.scatter(df, x="amount1", y="amount2",color="exclu",hover_data=[df.index])
    fig.show()


interetCommun(tab1,tab2)