import pandas as pd
from dash import Dash, html, dcc, Input, Output, State
import plotly.express as px
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import base64

vocab = ['france',
 'pays',
 'français',
 'président',
 'militaire',
 'état',
 'force',
 'armée',
 'américain',
 'afrique',
 'politique',
 'mali',
 'an',
 'ministre',
 'grand',
 'dernier',
 'terroriste',
 'gouvernement',
 'région',
 'national',
 'guerre',
 'occidental',
 'groupe',
 'africain',
 'malien',
 'russe',
 'iran',
 'russie',
 'attaque',
 'monde',
 'population',
 'sécurité',
 'affaire',
 'étranger',
 'nouveau',
 'pari',
 'sahel',
 'année',
 'homme',
 'européen',
 'accord',
 'opération',
 'macron',
 'jour',
 'média',
 'iranien',
 'israël',
 'international',
 'chef',
 'personne',
 'peuple',
 'général',
 'israélien',
 'place',
 'base',
 'source',
 'défense',
 'cas',
 'nord',
 'mort',
 'coup',
 'question',
 'jeune',
 'droit',
 'armé',
 'mois',
 'chine',
 'police',
 'projet',
 'million',
 'al',
 'algérie',
 'public',
 'avril',
 'arme',
 'janvier',
 'frontière',
 'zone',
 'situation',
 'soldat',
 'juin',
 'mai',
 'burkina',
 'social',
 'face',
 'fois',
 'présence',
 'ville',
 'système',
 'autorité',
 'femme',
 'algérien',
 'pouvoir',
 'civil',
 'syrie',
 'loi',
 'rapport',
 'missile',
 'manifestation',
 'syrien',
 'crise',
 'sputnik',
 'niger',
 'euro',
 'côte',
 'relation',
 'février',
 'juillet',
 'faso',
 'mars',
 'économique',
 'temps',
 'fin',
 'république',
 'ancien',
 'conseil',
 'territoire',
 'washington',
 'jaune',
 'bon',
 'santé',
 'information',
 'élection',
 'octobre',
 'plan',
 'service',
 'europe',
 'otan',
 'nucléaire',
 'résistance',
 'policier',
 'mesure',
 'mouvement',
 'trump',
 'gilet',
 'aérien',
 'occident',
 'sanction',
 'union',
 'occupation',
 'cameroun',
 'société',
 'chose',
 'conflit',
 'petit',
 'violence',
 'juif',
 'point',
 'ministère',
 'terrorisme',
 'drone',
 'lieu',
 'août',
 'septembre',
 'enfant',
 'partie',
 'éthiopie',
 'part',
 'décembre',
 'chinois',
 'coopération',
 'vaccin',
 'article',
 'membre',
 'sioniste',
 'avion',
 'milliard',
 'mission',
 'réseau',
 'novembre',
 'semaine',
 'lutte',
 'ivoire',
 'onu',
 'axe',
 'puissance',
 'ordre',
 'côté',
 'combat',
 'continent',
 'intérêt',
 'centre',
 'communauté',
 'parti',
 'décision',
 'journaliste',
 'turquie',
 'troupe',
 'irak',
 'mondial',
 'raison',
 'vie',
 'sanitaire',
 'heure',
 'liberté',
 'paix',
 'centrafricain',
 'éthiopien',
 'sénégal',
 'nation']

def from_csv(file_name):
    kw = pd.read_csv(file_name)
    kw["vector"] = kw["vector"].apply(lambda x : eval(x))
    return kw

def elbow_fitting(keywords):
    # distortions = []
    # K = range(1,15)
    # for k in K:
    #     kmeans = KMeans(n_clusters = k, random_state=0).fit(keywords["vector"].to_list())
    #     distortions.append(kmeans.inertia_)
    # return distortions

    return [2576722.6466701934,
 2283157.987564939,
 2189813.0864680493,
 2126115.2821893487,
 2086653.8712515705,
 2053759.5176914816,
 2021847.9820919218,
 2006874.2367311926,
 1976039.4456783538,
 1964826.0649371315,
 1936578.4848803026,
 1919869.517330286,
 1900896.1845776797,
 1875542.289194238]

def kmeans_clustering(keywords, n):
    kmeans = KMeans(n_clusters = n, random_state=0).fit(keywords["vector"].to_list())
    keywords["kmeans"] = list(kmeans.labels_)
    return keywords

def cluster_sizes(clusters):
    return clusters.groupby("kmeans")["Media"].count()

def get_cluster_keywords(clusters, n):
    cluster_keywords = pd.DataFrame([], columns= ["keyword", "count", "cluster"])
    for j in range(n):
        vector_summed = np.sum(np.asarray([ kw for kw in clusters[clusters["kmeans"] == j ]["vector"]]), axis = 0)
        kw_group = pd.DataFrame([[vocab[i], kw] for i, kw in enumerate(vector_summed) if kw > 0]).sort_values(1, ascending = False)
        kw_group = kw_group.head(20)
        kw_group.columns = ["keyword", "count"]
        kw_group["cluster"] = [j for i in range(20)]
        cluster_keywords = pd.concat([cluster_keywords, kw_group])
    return cluster_keywords
        
kw_pre_clustering = from_csv('kw_pre-clustering.csv')
n = 5

#print(cluster_sizes(clusters))

def generate_cluster_bar_chart(clusters, n):
    cluster_keywords = get_cluster_keywords(clusters, n)
    fig = px.bar(cluster_keywords, x='keyword', y='count',
        color='cluster',
       title=f"Most popular keywords by cluster (n = {n})")
    fig.update_layout( legend=dict(
        orientation="h",
        entrywidth=70,
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        bgcolor="LightSteelBlue",
    ))
    return fig


def generate_elbow_fitting_graph(kw_pre_clustering):
    distortions = elbow_fitting(kw_pre_clustering)
    indexes = range(1, len(distortions) + 1)
    elbow = pd.DataFrame(list(zip(distortions, indexes)), columns=["KMean distortion", "n"])
    fig = px.line(elbow, x="n", y="KMean distortion", title="Elbow fitting", markers=True)
    fig.update_layout(
        xaxis = dict(
            tickmode = 'linear',
            tick0 = 1,
            dtick = 1,
            gridcolor = "#d4d4d4"
        ), plot_bgcolor="white",
        yaxis = dict(gridcolor = "#d4d4d4")
        )
    return fig

def generate_cluster_count_bar_chart(clusters, n):
    sizes = cluster_sizes(clusters).to_frame()
    sizes.index.names = ["cluster"]
    sizes.columns = ["number of articles"]
    sizes = sizes.reset_index()
    return px.bar(sizes, x = "cluster", y ="number of articles", title= "Article count by cluster")

def generate_article_scatter_by_cluster(clusters, n):
    vectors = clusters["vector"].to_list()
    pca = PCA(n_components=2).fit(vectors)
    pca_2d = pca.transform(vectors)
    clusters["x"] = pca_2d[:,0]
    clusters["y"] = pca_2d[:,1]
    clusters = clusters.rename(columns = {'kmeans' : 'cluster'})
    clusters["cluster"] = clusters["cluster"].apply(lambda x : str(x))
    return px.scatter(clusters, x="x", y="y", color="cluster", title="2D PCA of articles colored by cluster")

def generate_article_scatter_by_media(kw_pre_clustering):
    vectors = kw_pre_clustering["vector"].to_list()
    pca = PCA(n_components=2).fit(vectors)
    pca_2d = pca.transform(vectors)
    kw_pre_clustering["x"] = pca_2d[:,0]
    kw_pre_clustering["y"] = pca_2d[:,1]
    return px.scatter(kw_pre_clustering, x="x", y="y", color="Media", title="2D PCA of articles colored by Media")

external_stylesheets = [
    {
        "href": "https://fonts.googleapis.com/css2?"
                "family=Lato:wght@400;700&display=swap",
        "rel": "stylesheet",
    },
]

app = Dash(__name__, external_stylesheets=external_stylesheets)


app.layout = html.Div(
    children=[
        html.Div(children = [
            html.H2(children="Clustering des articles", className ='header-title'),
            html.P(children="Clustering des articles du corpus à partir de leurs mots-clés à l'aide de la méthode KMeans.", className='header-description'),
            html.P(children=" Cette analyse permet de faire apparaitre différents thèmes au sein des articles du corpus selon le nombre n de"
                            " clusters choisi.", className='header-description', style={'font-style' : 'italic'})
        ], className='header'),
        
        html.Div(
            children=[
                
                html.H3(children = "Elbow Method"),
                html.P(children = "Pour choisir le nombre n de cluster optimal, on passe par la méthode Elbow qui consiste à plotter"
                    " la distortion de la méthode KMean pour plusieurs n.", style={'font-style' : 'italic'}),
                dcc.Graph(config={"displayModeBar": False}, figure = generate_elbow_fitting_graph(kw_pre_clustering), style={'width': '800px', 'height' : '600px'}, className='card'),

                html.P(children = "On cherche le n à partir duquel la courbe est décroissante de façon linéaire. Ici, aux alentours de n = 4.", style={'font-style' : 'italic'}),

                html.H3(children = "Clustering"),


                html.Div(children=[
                    html.Span(children="Number of clusters"),
                    dcc.Input(id="n", type="number", value=5),
                    html.Button('Calculate KMeans', id='submit', n_clicks=0)
                ]),

                html.P(children = "Expérimentalement, n = 5 donne un clustering pertinent." 
                " On peut essayer d'augmenter n pour identifier plus de sous-thèmes parmi les articles.", style={'font-style' : 'italic'}),
                
                html.Br(),

                dcc.Graph(id = 'cluster-count', style={'width': '500px', 'height' : '500px'}, className="card"),
                dcc.Graph(id = 'cluster-chart', className="card-big", style={'width': '80vw', 'height' : '600px'}),
                html.P(children = "Avec n = 5, on observe que les thèmes principaux sont la France, la politique, les États-Unis, l'Afrique et le Mali, le domaine militaire et les juifs." 
                , style={'font-style' : 'italic'}),

                html.Div(
                    children = [
                        dcc.Graph(id = 'article-clusters',style={'width': '30vw', 'height' : '600px', 'display' : 'inline-block'}, className='card'),
                        dcc.Graph(id = 'article-medias', style={'width': '38vw', 'height' : '600px', 'display' : 'inline-block'}, className='card')
                    ]
                ),

                html.P(children = "Avec la Principal Component Analysis 2D, on observe que le groupement par clusters n'est pas identique au groupement des articles par médias." 
                " Cela signifie qu'on ne peut pas distinguer les sources en analysant les mots-clés et les thèmes qui en ressortent, et donc que chaque journal peut traiter de thèmes communs.", style={'font-style' : 'italic'}),

                #html.Iframe(src="assets/cluster_200kw.svg", style={'width' : "80%", 'height':"800px"}),
                html.Div(children=[
                    html.Img(src="assets/cluster_200kw.svg")
                ], style={'width' : "80%", 'height':"800px", 'overflow' : 'scroll', 'background-color' : 'black'}),

                html.Iframe(src="assets/network/index.html", style={'width' : "80%", 'height':"800px"})

            ], className='wrapper'
        )
    ]
)

@app.callback(
    Output("cluster-count", "figure"),
    Output("cluster-chart", "figure"),
    Output("article-clusters", "figure"),
    Output("article-medias", "figure"),
    Input("submit", "n_clicks"),
    State("n", "value")
)
def update_kmeans(n_click, n):
    clusters = kmeans_clustering(kw_pre_clustering, n)
    return generate_cluster_count_bar_chart(clusters, n), generate_cluster_bar_chart(clusters, n), generate_article_scatter_by_cluster(clusters, n), generate_article_scatter_by_media(kw_pre_clustering)



if __name__ == "__main__":
    app.run_server(debug=True)