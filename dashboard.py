import pandas as pd
from dash import Dash, html, dcc, Input, Output, State
import plotly.express as px
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import dash_dangerously_set_inner_html
import math

medias = ['french.presstv.ir', 'www.egaliteetreconciliation.fr',
       'www.fdesouche.com', 'fr.sputniknews.africa']


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
    
def from_csv_occurences(file_name):
    occ = pd.read_csv(file_name)
    for source in ["total", *medias]:
        occ[source] = occ[source].apply(lambda x : eval(x.replace(" ", "").replace(".", ",")))
    return occ

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


vocab500 = ['france',
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
 'nation',
 'aide',
 'vidéo',
 'zemmour',
 'coronavirus',
 'turc',
 'enquête',
 'manifestant',
 'problème',
 'suite',
 'régime',
 'tchad',
 'moyen',
 'navire',
 'allemand',
 'noir',
 'émission',
 'arabe',
 'menace',
 'nombre',
 'presse',
 'libye',
 'important',
 'organisation',
 'exercice',
 'camerounais',
 'droite',
 'site',
 'sud',
 'communiqué',
 'propos',
 'compte',
 'ambassadeur',
 'allié',
 'daech',
 'musulman',
 'cadre',
 'guinée',
 'responsable',
 'migrant',
 'allemagne',
 'islamique',
 'air',
 'mainstream',
 'virus',
 'palestinien',
 'dollar',
 'visite',
 'marine',
 'entreprise',
 'vaccination',
 'contrôle',
 'moscou',
 'histoire',
 'soudan',
 'alain',
 'ivoirien',
 'pression',
 'britannique',
 'ouest',
 'soutien',
 'justice',
 'irakien',
 'cause',
 'transition',
 'golfe',
 'développement',
 'famille',
 'blanc',
 'travail',
 'gauche',
 'action',
 'maroc',
 'association',
 'victime',
 'moment',
 'mer',
 'saoudien',
 'nigérien',
 'cours',
 'journal',
 'rebelle',
 'haine',
 'économie',
 'alliance',
 'exemple',
 'pétrolier',
 'vrai',
 'stratégique',
 'soral',
 'terme',
 'main',
 'emmanuel',
 'liban',
 'niveau',
 'intérieur',
 'tentative',
 'françois',
 'masque',
 'marché',
 'sommet',
 'position',
 'député',
 'renseignement',
 'port',
 'attentat',
 'hezbollah',
 'us',
 'nom',
 'local',
 'libanais',
 'tchadien',
 'djihadiste',
 'alger',
 'ambassade',
 'eau',
 'pétrole',
 'école',
 'banque',
 'rdc',
 'opposition',
 'lundi',
 'samedi',
 'maire',
 'étude',
 'afghanistan',
 'objectif',
 'administration',
 'sécuritaire',
 'coalition',
 'province',
 'avocat',
 'arabie',
 'appel',
 'production',
 'début',
 'premier',
 'livre',
 'barkhan',
 'gaz',
 'naval',
 'acte',
 'sein',
 'discours',
 'haut',
 'photo',
 'campagne',
 'barkhane',
 'ennemi',
 'retrait',
 'sénégalais',
 'village',
 'candidat',
 'rue',
 'épidémie',
 'bamako',
 'manière',
 'islam',
 'major',
 'mercredi',
 'souveraineté',
 'fille',
 'frappe',
 'prison',
 'agence',
 'besoin',
 'libre',
 'mardi',
 'résultat',
 'téhéran',
 'mise',
 'camp',
 'séparatiste',
 'culture',
 'racisme',
 'déstabilisation',
 'commission',
 'ue',
 'poutine',
 'prix',
 'image',
 'programme',
 'présidentiel',
 'financier',
 'chaîne',
 'réunion',
 'but',
 'ong',
 'révolution',
 'dirigeant',
 'royaume',
 'quartier',
 'agent',
 'capacité',
 'venezuela',
 'feu',
 'vendredi',
 'risque',
 'populaire',
 'activité',
 'terre',
 'habitant',
 'vol',
 'confinement',
 'libération',
 'échec',
 'maritime',
 'tigré',
 'produit',
 'hôpital',
 'libyen',
 'expert',
 'incendie',
 'long',
 'ligne',
 'tête',
 'antisémitisme',
 'marocain',
 'véhicule',
 'liste',
 'sens',
 'commandant',
 'dialogue',
 'scientifique',
 'total',
 'congolais',
 'unité',
 'cour',
 'assassinat',
 'faux',
 'second',
 'burkinabé',
 'formation',
 'meilleur',
 'médecin',
 'propre',
 'stratégie',
 'message',
 'centrafriqu',
 'yémen',
 'actuel',
 'kilomètre',
 'citoyen',
 'document',
 'idée',
 'négociation',
 'tension',
 'crime',
 'parisien',
 'jeudi',
 'sujet',
 'front',
 'intervention',
 'tribunal',
 'régional',
 'équipe',
 'dimanche',
 'september',
 'travers',
 'secrétaire',
 'poste',
 'secteur',
 'mandat',
 'département',
 'lien',
 'donnée',
 'principal',
 'uni',
 'débat',
 'august',
 'officiel',
 'initiative',
 'acteur',
 'humain',
 'humanitaire',
 'numéro',
 'processus',
 'maison',
 'œuvre',
 'auteur',
 'éric',
 'solution',
 'spécial',
 'mozambique',
 'argent',
 'vue',
 'antisémite',
 'extrême',
 'maladie',
 'déclaration',
 'contrat',
 'explosion',
 'conférence',
 'compagnie',
 'religion',
 'plainte',
 'réalité',
 'afghan',
 'agression',
 'gazer',
 'départ']

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


def generate_plot_evolution(kw, source, occurences):
  i = vocab500.index(kw)
  evolution = occurences.copy()
  evolution['date'] = pd.to_datetime(evolution['date'], format='%Y/%m/%d')

  for s in ["total", *medias]:
    evolution[s] = evolution[s].apply(lambda x : x[i])
    
  fig = make_subplots(
    rows=len(["total", *medias]), cols=1,
    subplot_titles=["total", *medias])
  
  for i, s in enumerate(["total", *medias]):
    fig.add_trace(go.Scatter(x=evolution["date"], y=evolution[s]),
              row=i+1, col=1)

  #fig = px.line(evolution, x="date", y=["total", *medias], title='occurence of keyword : '+ kw + ", source : " + source)
  fig.update_layout(height=950, width=1200, title_text="occurence of keyword : "+ kw)
  return fig

def generate_plot_correlation(kw, occurences):
  i = vocab500.index(kw)
  evolution = occurences.copy()
  evolution["total"] = evolution["total"].apply(lambda x : x[i])
  correlations = []

  for kw_corr in vocab500:
    if kw_corr == kw:
      continue
    evolution2 = occurences.copy()
    j = vocab500.index(kw_corr)
    evolution2["total"] = evolution2["total"].apply(lambda x : x[j])
    corr = evolution["total"].corr(evolution2["total"])
    correlations.append([kw_corr, corr])
  
  correlations = pd.DataFrame(correlations)
  correlations.columns = ["keyword", "correlation"]
  correlations = correlations.sort_values("keyword")
  top10 = correlations.sort_values("correlation", ascending = False).head(10)
  correlations["Significance"] = correlations["keyword"].apply(lambda x : 1 if x in top10["keyword"].to_numpy() else 0.5)
  fig = px.scatter(correlations, x="keyword", y="correlation", title='Correlations pour : '+ kw, color="Significance", color_continuous_scale="earth")
  fig.update(layout_coloraxis_showscale=False)
  top10 = top10.set_index("keyword")
  return fig, top10.to_html()


occurences = from_csv_occurences("occurences.csv")

def get_top_kw_from_source(source, occurences, n):
  total_source = np.sum(np.asarray([ occs for occs in occurences[source]]), axis = 0)
  total_source = pd.DataFrame(list(zip(vocab500, total_source)))
  total_source.columns = ["keyword", "amount"]
  total_source["percentage"] = total_source["amount"].apply(lambda x : x/total_source["amount"].sum()*100)
  total_source = total_source.sort_values("amount", ascending = False).head(n)
  return total_source

def common_interest(source1, source2, n):
  df1 = get_top_kw_from_source(source1, occurences, n)
  df2 = get_top_kw_from_source(source2, occurences, n)
  all_keywords = pd.concat([df1["keyword"], df2["keyword"]]).drop_duplicates()
  compared_interest = []
  for kw in all_keywords:
    row = [kw]
    try:
      if not len(df1.loc[df1["keyword"] == kw]["amount"].index):
        raise KeyError

      amount = df1.loc[df1["keyword"] == kw]["amount"].iloc[0]
      percentage = df1.loc[df1["keyword"] == kw]["percentage"].iloc[0]
      row = [*row, amount, percentage]
    except KeyError:
      row = [*row, 0, 0]
    try:
      if not len(df2.loc[df2["keyword"] == kw]["amount"].index):
        raise KeyError

      amount = df2.loc[df2["keyword"] == kw]["amount"].iloc[0] 
      percentage = df2.loc[df2["keyword"] == kw]["percentage"].iloc[0]
      row = [*row, amount, percentage]
    except KeyError:
      row = [*row, 0, 0]

    compared_interest.append(row)
  compared_interest = pd.DataFrame(compared_interest)
  compared_interest.columns = ["keyword", "amount_1", "percentage_1", "amount_2", "percentage_2" ]


  def shortest_distance(x1, y1, a, b, c):
    d = abs((a * x1 + b * y1 + c)) / (math.sqrt(a * a + b * b))
    d = -d if x1 > y1 else d
    return d

  compared_interest["distance"] = compared_interest.apply(lambda x: shortest_distance(x["percentage_1"], x["percentage_2"], 1, -1, 0 ), axis = 1)
  compared_interest = compared_interest.sort_values("distance")


  fig=px.scatter(compared_interest, x="keyword", y="distance",color="distance",title ="Facteur d'intêret commun par mot-clé" , labels={
                     "keyword": "keyword",
                     "distance": "Facteur d'intérêt commun",
                 },hover_data=["keyword", "amount_1",  "amount_2", "percentage_1", "percentage_2", "distance"], color_continuous_scale=["red", "grey", "blue"], color_continuous_midpoint = 0)
  fig.update_yaxes(
    scaleanchor = "x",
    scaleratio = 1,
  )
  fig.update_layout(coloraxis_colorbar=dict(
    title="Ligne editoriale",
    tickvals=[compared_interest["distance"].min(),compared_interest["distance"].max()],
    ticktext=[source1, source2],
))

  random_x = np.linspace(0, 4, 5)
  fig2=px.scatter(compared_interest, x="percentage_1", y="percentage_2",color="distance",hover_data=["keyword", "amount_1",  "amount_2", "percentage_1", "percentage_2", "distance"],
                 color_continuous_scale=["red", "grey", "blue"], color_continuous_midpoint = 0, labels={
                     "percentage_1": "percentage source 1",
                     "percentage_2": "percentage source 2",
                 }, title="Comparaison de la proportion d'utilisation des mots clés")
                 
  fig2.add_trace(go.Scatter(x=random_x, y=random_x,
                    mode='lines',
                    name='y=x'))

  fig2.update_layout(coloraxis_colorbar=dict(
    title="Ligne Editoriale",
    tickvals=[compared_interest["distance"].min(),compared_interest["distance"].max()],
    ticktext=[source1, source2],
))

  return fig, fig2


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

                html.Div(children=[
                    
                    html.P(children = "Pour choisir le nombre n de cluster optimal, on passe par la méthode Elbow qui consiste à plotter"
                        " la distortion de la méthode KMean pour plusieurs n.", style={'font-style' : 'italic'}),
                    dcc.Graph(config={"displayModeBar": False}, figure = generate_elbow_fitting_graph(kw_pre_clustering), style={'width': '800px', 'height' : '600px'}),

                    html.P(children = "On cherche le n à partir duquel la courbe est décroissante de façon linéaire. Ici, aux alentours de n = 4.", style={'font-style' : 'italic'})
                ], className='card'),
                
                

                html.H3(children = "Clustering"),

                html.Div(children = [
                    html.P(children=[
                        html.Span(children="Number of clusters :"),
                        dcc.Input(id="n", type="number", value=5, style={"margin-left" : '10px'}),
                        html.Button('Compute KMeans', id='submit', n_clicks=0),
                        html.Div(children = "Expérimentalement, n = 5 donne un clustering pertinent." 
                    " On peut essayer d'augmenter n pour identifier plus de sous-thèmes parmi les articles.", style={'font-style' : 'italic', "margin-top" : '15px'}),
                    ]),
                    html.Br(),
                    dcc.Graph(id = 'cluster-count', style={'width': '500px', 'height' : '500px', 'margin' : '0 auto'}),
                ], className="card"),

                html.Div(children = [
                    html.P(children = 
                    " Cliquer sur la légende pour filtrer par cluster."
                    , style={'font-style' : 'italic'}),
                    dcc.Graph(id = 'cluster-chart', style={'width': '80vw', 'height' : '600px'}),
                    html.P(children = "Avec n = 5, on observe que les thèmes principaux sont la France, la politique, les États-Unis, l'Afrique et le Mali, le domaine militaire et les juifs. "
                    , style={'font-style' : 'italic'}),
                ], className="card-big"),
                
                html.Div(
                    children = [
                       html.Div(
                    children = [
                        dcc.Graph(id = 'article-clusters',style={'width': '38vw', 'height' : '600px', 'display' : 'inline-block'}),
                        dcc.Graph(id = 'article-medias', style={'width': '42vw', 'height' : '600px', 'display' : 'inline-block'})
                    ]
                ),

                html.P(children = "En projetant le vecteur de keywords associé à chaque article avec la Principal Component Analysis 2D, on observe que le groupement par clusters n'est pas identique au groupement des articles par médias." 
                " Cela signifie qu'on ne peut pas distinguer les sources en analysant les mots-clés et les thèmes qui en ressortent, et donc que chaque journal peut traiter de thèmes communs.", style={'font-style' : 'italic'}),
                    ], className="card-big"
                )
            ], className='wrapper'
        ),

        html.Div(children = [
            html.H2(children="Clustering des mots-clés", className ='header-title'),
            html.P(children="Clustering des mots-clés du corpus à partir d'un graphe de co-occurence", className='header-description'),
            html.P(children=" Cette analyse permet de faire apparaitre différents thèmes au sein des mots-clés du corpus.", className='header-description', style={'font-style' : 'italic'})
            ], className='header'),

        html.Div(children=[
            html.Div(children =[
                html.P(children=["A partir d'une matrice de co-occurence des 200 keywords les plus fréquents, on produit un graphe pondéré non-dirigé des mots-clés à l'aide de l'outil Gephi.",
                    html.Br(),
                    "On réalise ensuite un clustering par modularité qui nous donne le graph suivant."
                ]),
                html.Div(children=[
                    html.Img(src="assets/Untitled.svg")
                ], style={'width' : "80%", 'height':"800px", 'overflow' : 'scroll', 'background-color' : 'black', 'margin' : "0 auto"}),
                html.P(children=["Les clusters obtenus regroupent les mot-clés qui apparaissent souvent ensemble et ont des liens similaires avec les autres clusters.",
                ]),
                 html.P(children=["Un autre layout permet de mettre en avant les mots-clés qui occurent le plus fréquement avec d'autres mots-clés.",
                ]),
                html.Div(children=[
                    html.Img(src="assets/graphdegree.svg")
                ], style={'width' : "80%", 'height':"800px", 'overflow' : 'scroll', 'background-color' : 'black', 'margin' : "0 auto"}),
            ], className='card-big'),
            

                html.H3(children = "Graph explorable"),
                html.Iframe(src="assets/network/index.html", style={'width' : "80%", 'height':"800px"})

        ], className='wrapper'),

        html.Div(children = [
            html.H2(children="Évolution des mots-clés", className ='header-title'),
            html.P(children="Analyse de l'évolution de l'occurence des mots-clés au cours tu temps.", className='header-description'),
            html.P(children=" Cette analyse permet d'étudier l'apparition des mots-clés au fil du temps et la correlation entre plusieurs mots-clés.", className='header-description', style={'font-style' : 'italic'})
            ], className='header'),

        html.Div(children=[
            html.Div(children =[
                html.P(children=["Analyse sur les 500 mots-clés les plus fréquents"]),
                dcc.Dropdown(sorted(vocab500), "zemmour", id ="kw-select"),
                html.Button('Plot evolution', id='submit-evol', n_clicks=0),
                dcc.Graph(id = 'kw-evol', style={'width': '80vw', 'height' : '950x', 'margin' : '0 auto'}),
                html.H3(children="Analyse de corrélation"),
                dcc.Graph(id="kw-corr",style={'width': '80vw', 'height' : '950x', 'margin' : '0 auto'}),
                html.Iframe(id='top10', style={'height': '375px', 'width' :'200px', 'margin' : '0 auto' }),
                html.P(children=["Cette analyse montre le coefficient de correlation de l'évolution mot-clé choisi avec les évolutions des autres mots-clés. Pour chaque autre mot-clés,"
                " on obtient un coefficient entre -1 et 1. Plus le coefficient est grand, plus il existe une corrélation linéaire positive entre les deux évolutions, ce qui signifie"
                " que lorsque l'une augmente l'autre aussi et idem pour une baisse."]),
                html.P(children=["Cette visualisation permet de remarquer les mots-clés qui apparaisent ensembles sur les mêmes périodes."])

            ], className='card-big'),
            
        ], className='wrapper'),

html.Div(children = [
            html.H2(children="Étude comparative des lignes éditoriales", className ='header-title'),
            html.P(children="Analyse comparative de l'utilisation des mots-clés", className='header-description'),
            html.P(children=" Cette analyse permet d'étudier la convergence et divergence de l'utilisation des mots-clés par différentes sources.", className='header-description', style={'font-style' : 'italic'})
            ], className='header'),
        html.Div(children=[


        html.Div(children =[

            html.Span(children = ["Source 1 :",
                dcc.Dropdown(medias, "french.presstv.ir", id ="source1")
            ]),

            html.Span(children =[
                "Source 2 :",
                 dcc.Dropdown(medias, "www.egaliteetreconciliation.fr", id ="source2")
            ]),

            html.Span(children =[
                "n mots clés les plus fréquents/source",
                 dcc.Input(id="n-ic", type="number", value=150, style={"margin-left" : '10px'})
            ]),


            html.Button('Plot editorial policy comparison', id='submit-ic', n_clicks=0),
            dcc.Graph(id = 'common-interest2', style={'width': '80vw', 'height' : '950x', 'margin' : '0 auto'}),
            dcc.Graph(id = 'common-interest', style={'width': '80vw', 'height' : '950x', 'margin' : '0 auto'}),
            html.P(children="Ce graphique montre pour chaque sources l'usage commun ou non des n mots clés les plus fréquents pour les deux sources. "
            " On calcule un facteur d'intérêt commun pour chaque mot-clés à partir du nombre d'occurences pour les deux sources, qui correspond à la distance entre "
            " le point (nb occurences source 1, nb occurences source2) et la droite y=x. Ainsi, une valeur autour de 0 signifie que le mot clé est employé à proportions égales par les "
            " deux sources. Une valeur significativement supérieur à 0 signifie qu'il est plus employé par la source 2, et inversement pour une valeur inférieure à 0.")
            

        ], className='card-big')

        ], className='wrapper')
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


@app.callback(
    Output("kw-evol", "figure"),
    Output('kw-corr', 'figure'),
    Output('top10', 'srcDoc'),
    Input("submit-evol", "n_clicks"),
    State("kw-select", "value"),
)
def update_evol(click, value):
    corr = generate_plot_correlation(value, occurences)
    return generate_plot_evolution(value, "total", occurences), corr[0], "<h2>Top 10</h2>" + corr[1]

@app.callback(
    Output("common-interest", "figure"),
    Output("common-interest2", "figure"),
    Input("submit-ic", "n_clicks"),
    State("source1", "value"),
    State("source2", "value"),
    State("n-ic", "value")
)
def update_common_interest(click, source1, source2,n):
    ci1, ci2 = common_interest(source1, source2, n)
    return ci1, ci2

if __name__ == "__main__":
    app.run_server(debug=True)