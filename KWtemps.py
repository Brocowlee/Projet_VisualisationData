import pandas as pd
import plotly.express as px
import json
from dash import Dash, dcc, html, Input, Output
import os

files_name = ['C:/Users/arthu/OneDrive/Documents/Travail/FI4/Visualisation/projMedia/preprocess-topaz-data732//' + file_name for file_name in os.listdir('C:/Users/arthu/OneDrive/Documents/Travail/FI4/Visualisation/projMedia/preprocess-topaz-data732//') if '.json' in file_name]

#months = {'1' : 0, '1' : 0, '2' : 0, '3' : 0, '4' : 0,'5' : 0, '6' : 0,'7' : 0,'8' : 0,'9' : 0,'10' : 0,'11' : 0,'12' : 0}
#for file_name in files_name:
#    print(file_name)
#    f = open(file_name, 'r', encoding='utf-8')
#    data = json.loads(f.read())
#    for year in data['data-all'].values():
#        for month in year.items():
#            for day in month[1].values():
#                months[month[0]] += len(day)
    
#months = {}
#for file_name in files_name:
#    print(file_name)
#    f = open(file_name, 'r', encoding='utf-8')
#    data = json.loads(f.read())
#    for year in data['data-all'].items():
#        for month in year[1].items():
#            print(f"{year[0]}-{month[0]}")
#            for day in month[1].values():
#                try:
#                    months[f"{year[0]}-{month[0]}"] += len(day) 
#                except:
#                    months[f"{year[0]}-{month[0]}"] = len(day)
#                
#def sort_key(x):
#    year, month = x[0].split("-")
#    return int(year) + int(month)
#
#months = dict(sorted(months.items(), key = sort_key))
#                
#fig = px.bar(pd.DataFrame(list(zip(months.keys(), months.values())), columns=['Month', "Amount"]), x='Month', y='Amount')
#fig.show()
#print(pd.DataFrame(list(zip(months.keys(), months.values())), columns=['Month', "Amount"]))
#fig.write_html("exememple.html")               
#    


# allKw=[]
# for file_name in files_name:
#     keywords_count = []
#     print(file_name)
#     f = open(file_name, 'r', encoding='utf-8')
#     data = json.loads(f.read())
#     for fr in data['metadata-all'].values():
#         for year in fr["year"].items():
#             for keyword in year[1]["kws"].items():
#                 keywords_count.append( [str(keyword[0]), int(keyword[1]),int(year[0]) ])
            
#     df = pd.DataFrame.from_dict(keywords_count).rename(columns = {0:"kw",1:"amount"})
#     df = df.groupby(["kw"]).mean()
#     df= df.nlargest(100,
    
def evolKW(kwRef,journal):
    df=pd.DataFrame(columns=["kw","amount","date"])
    f = open(journal, 'r', encoding='utf-8')
    data = json.loads(f.read())
    counter=0
    for fr in data['metadata-all'].values():
        for year in fr["day"].items():
            for month in year[1].items():
                for day in month[1].items():
                    for kw in day[1]["kws"].items():
                        if str(kw[0])==kwRef:
                            if len(str(month[0]))==1:
                                df.loc[counter]=[str(kw[0]),str(kw[1]),str(year[0])+"/0"+str(month[0])+"/"+str(day[0])]
                            if len(str(day[0]))==1:
                                df.loc[counter]=[str(kw[0]),str(kw[1]),str(year[0])+"/"+str(month[0])+"/0"+str(day[0])]
                            if len(str(day[0]))==1 and len(str(month[0]))==1:
                                df.loc[counter]=[str(kw[0]),str(kw[1]),str(year[0])+"/0"+str(month[0])+"/0"+str(day[0])]
                            df.loc[counter]=[str(kw[0]),int(kw[1]),str(year[0])+"/"+str(month[0])+"/"+str(day[0])]

                            counter+=1

    df=df.set_index("date")
    df=df.sort_index()
    print(df.head(20))
    fig = px.line(df, x=df.index, y="amount", title='presence of the world '+kwRef)
    fig.show()


evolKW("attentat", "C:/Users/arthu/OneDrive/Documents/Travail/FI4/Visualisation/projMedia/preprocess-topaz-data732/topaz-data732--mali--www.egaliteetreconciliation.fr--20190101--20211231.json")


# app = Dash(__name__)

# app.layout = html.Div([
#     html.H4('10 Most recurrent keyword'),
#     dcc.Dropdown(
#         id="dropdown",
#         options= [2018, 2019, 2020, 2021],
#         value=[2018],
#         clearable=False,
#     ),
#     dcc.Graph(id="graph"),
# ])


# @app.callback(
#     Output("graph", "figure"), 
#     Input("dropdown", "value"))
# def update_bar_chart(year):
#     df1 = df # replace with your own data source
#     mask = df1["year"] == year
#     fig = px.bar(df1[mask], x="kw", y="amount", 
#                  color="kw", barmode="group")
#     return fig


# app.run_server(debug=True)