#Importation des librairies MAJ le 13/01/2023
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import datetime
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn import cluster, metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
from sklearn.metrics import silhouette_samples, silhouette_score

def data_analyse(df):
    """
    Permet de réaliser une analyse des données (infos, outliers, valeurs manquantes et doublons)
    """
    print("Infos du dataframe:")
    print("-"*20)
    print(df.info())
    print("\n")
    print("Vérification des outliers:")
    print("-"*20)
    print(df.describe())
    print("\n")
    print("Valeurs manquantes:")
    print("-"*20)
    print(df.isnull().sum())
    print("\n")
    print("% Valeurs manquantes:")
    print("-"*20)
    print(df.isna().mean())
    print("\n")
    print("Vérification des doublons:")
    print("-"*20)
    print(df[df.duplicated()].head())

def test_cle(df,columns):
    """
    Permet de vérifier l'unicité d'une clé primaire
    """
    a=df.drop_duplicates(subset=columns).shape[0]
    b=df.shape[0]
    if a==b: print("La clé est unique")
    else: print("La clé n'est pas unique")

def find_outliers_iqr(data):
    """
    Permet de rechercher des outliers via la méthode de la distance interquartile
    """
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    return outliers

def find_outliers(v):
    """
    Permet de rechercher des outliers via la méthode de la distance interquartile
    """
    Q1 = np.quantile(v, 0.25)
    Q3 = np.quantile(v,0.75)
    EIQ = Q3 - Q1
    LI = round(Q1 - (EIQ*1.5),2)
    print("La limite inférieure est de",LI,)
    LS = round(Q3 + (EIQ*1.5),2)
    print("La limite supérieure est de",LS,)
    print("Liste des outliers ---")
    print(v.loc[(v < LI) | (v > LS)].to_frame())

def find_outliers_zscore(data):
    """
    Permet de rechercher des outliers via la méthode du Z-Score
    """
    outliers = []
    threshold=3
    mean=np.mean(data)
    std=np.std(data)
    
    for i in data:
        z_score=(i-mean) / std
        if np.abs(z_score)>threshold:
            outliers.append(i)
    return outliers

def etat_jointure(df,columns):
    """
    Permet de faire le point sur l'état d'une jointure
    """
    return df.value_counts(columns)

def pricing_analysis(data):
    """
    Permet de réaliser une analyse univariée des prix
    """
    print(data.describe())
    print("\n")
    print(data.mode())
    print("\n")
    print("Skewness empirique =",round(data.skew(),2),)
    print("\n")
    plt.figure(figsize=(16,5))
    plt.hist(data,ec="black")
    plt.title("Distribution empirique des prix",fontsize=12)
    plt.xlabel("Prix", fontsize=10)
    plt.ylabel("Count", fontsize=10)
    plt.show()
    print("\n")
    plt.figure(figsize=(16,5))
    plt.boxplot(data, vert=False)
    plt.show()
    find_outliers(data)
    return

def make_autopct(values):
    """
    Permet d'afficher les valeurs sur un pie chart
    """
    def my_autopct(pct):
        total = sum (values)
        val = int(round(pct*total/100.0))
        return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
    return my_autopct

def lorenz_gini(data,title):
    """
    Permet de faire une courbe de Lorenz et un indice de Gini
    """
    x = data.values
    n = len(x)
    lorenz = np.cumsum(np.sort(x)) / x.sum()
    lorenz = np.append([0],lorenz)
    xaxis = np.linspace(0-1/n,1+1/n,n+1)
    plt.plot(xaxis,lorenz,drawstyle='steps-post')
    plt.plot([0,1], [0,1])
    plt.title(title)
    plt.show()
    AUC = (lorenz.sum() -lorenz[-1]/2 -lorenz[0]/2)/n
    S = 0.5 - AUC
    gini = 2*S
    print("Indice de Gini =",round(gini,2))
    return

def tendance_centrale(data):
    """
    Mesure de tendance centrale
    """
    print("Mesures de tendance centrale:")
    print("-"*20)
    print("Mode =",data.mode())
    print("Moyenne =",round(data.mean(),2))
    print("Médiane =",round(data.median(),2))

def dispersion(data):
    """
    Mesure de dispersion
    """
    print("Mesures de dispersion:")
    print("-"*20)
    print("Variance empirique =",round(data.var(),2))
    print("Variance empirique sans biais =",round(data.var(ddof=0),2))
    print("Ecart-Type empirique =",round(data.std(),2))
    print("Coéfficient de variation =",round(data.std(ddof=0),2))
    plt.figure(figsize=(16,5))
    plt.boxplot(data, vert=False)
    plt.show()
    return

def forme(data):
    """
    Mesure de forme
    """
    print("Mesures de forme:")
    print("-"*20)
    print("Skewness empirique =",round(data.skew(),2))
    print("Kurtosis empirique =",round(data.kurtosis(),2))

def distribution(data,title):
    """
    Distribution des variables
    """
    plt.figure(figsize=(16,5))
    plt.hist(data,ec="black")
    plt.title(title,fontsize=12)
    plt.show()
    return

def analyse_univariee(data,title_distribution,title_lorenz):
    """
    Permet de réaliser une analyse univariée sur un df
    """
    plt.figure(figsize=(16,5))
    plt.hist(data,ec="black")
    plt.title(title_distribution,fontsize=12)
    plt.show()
    print("\n")
    print("Mesures de tendance centrale:")
    print("-"*20)
    print("Mode =",data.mode())
    print("Moyenne =",round(data.mean(),2))
    print("Médiane =",round(data.median(),2))
    print("\n")
    print("Mesures de dispersion:")
    print("-"*20)
    print("Variance empirique =",round(data.var(),2))
    print("Variance empirique sans biais =",round(data.var(ddof=0),2))
    print("Ecart-Type empirique =",round(data.std(),2))
    print("Coéfficient de variation =",round(data.std(ddof=0),2))
    plt.figure(figsize=(16,5))
    plt.boxplot(data, vert=False)
    plt.show()
    print("\n")
    print("Mesures de forme:")
    print("-"*20)
    print("Skewness empirique =",round(data.skew(),2))
    print("Kurtosis empirique =",round(data.kurtosis(),2))
    print("\n")
    print("Mesures de concentration:")
    print("-"*20)
    x = data.values
    n = len(x)
    lorenz = np.cumsum(np.sort(x)) / x.sum()
    lorenz = np.append([0],lorenz)
    xaxis = np.linspace(0-1/n,1+1/n,n+1)
    plt.plot(xaxis,lorenz,drawstyle='steps-post')
    plt.plot([0,1], [0,1])
    plt.title(title_lorenz)
    plt.show()
    AUC = (lorenz.sum() -lorenz[-1]/2 -lorenz[0]/2)/n
    S = 0.5 - AUC
    gini = 2*S
    print("Indice de Gini =",round(gini,2))
    return

def perform_hac(data,nb_clusters_den=10,nb_clusters=3):
    """
    Permet de faire une CLASSIFICATION ASCENDANTE HIERARCHIQUE (CAH) à partir des données en utilisant la méthode de Ward
    """
    df_cah = data.copy()
    # Calcul de la matrice des distances
    Z = linkage(df_cah, method="ward")
    # Tracez le dendrogramme
    plt.figure(figsize=(16,5))
    dendrogram(Z, p=nb_clusters_den,truncate_mode="lastp")
    plt.ylabel("Distance")
    plt.title("Hierarchical Clustering Dendrogram", fontsize=18)
    plt.show()
    # Définition des clusters
    cah = AgglomerativeClustering(n_clusters=nb_clusters, linkage="ward")
    cah.fit(df_cah)
    labels = cah.labels_
    # Création d'une colonne "Cluster"
    df_cah["Cluster"] = labels
    return df_cah

def elbow_method(data, max_clusters):
    """
    Appliquer la méthode du coude pour déterminer le nombre optimal de clusters.
    """
    # Création d'une liste vide pour stocker les inerties
    inertie = []
    # Ajuster et prédire KMeans pour chaque nombre de clusters
    for i in range(1, max_clusters+1):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(data)
        inertie.append(kmeans.inertia_)
    # Création du graphique pour afficher le résultat
    plt.figure(figsize=(12,6))
    plt.plot(range(1, max_clusters+1), inertie, marker = "o")
    plt.title('Méthode du coude')
    plt.xlabel('Nombre de clusters')
    plt.ylabel('Inertie')
    plt.show()

def perform_kmeans(data, n_clusters=3):
    """
    Permet de faire un clustering via l'algorithme k-means à partir des données
    """
    # Initialisation de l'algorithme KMeans
    kmeans = KMeans(n_clusters=n_clusters)
    # Ajustement des données
    kmeans.fit(data)
    # Définition des clusters
    cluster_labels = kmeans.labels_
    # Définition des centroides (centre de gravité des clusters)
    centroids = kmeans.cluster_centers_
    # Création d'une colonne "Cluster"
    data["Cluster"] = cluster_labels
    return data

def plot_silhouette_score(data):
    """
    Tracer une courbe de Silhouette pour les données
    """
    #Créez une liste de scénarios hypothétiques pour différents nombres de clusters
    kmeans_per_k = [KMeans(n_clusters=k,random_state=42).fit(data) for k in range (1,10)]
    silhouette_scores = [silhouette_score(data,model.labels_) for model in kmeans_per_k[1:]]
    k= np.argmax(silhouette_scores) + 2
    # Création du graphique
    plt.figure(figsize=(12,6))
    plt.plot(range(2,10), silhouette_scores, marker="+", color="blue", linewidth=2, markersize=8, label='Courbe de Silhouette')
    plt.xlabel("Nombre de cluster")
    plt.ylabel("Silhouette Score")
    plt.title("Courbe de Silhouette pour prédire le nombre optimal de clusters")
    plt.axvline(x=k, linestyle='--', linewidth=2, label='Nombre optimal de clusters ({})'.format(k))
    plt.legend(shadow=True)
    plt.show()
    print('Le nombre optimal de cluster est {}.'.format(k))
    return

def cercle_correlation(pca, x_y, features) : 
    """Affiche le cercle des correlations

    Positional arguments : 
    -----------------------------------
    pca : sklearn.decomposition.PCA : notre objet PCA qui a été fit
    x_y : list ou tuple : le couple x,y des plans à afficher, exemple [0,1] pour F1, F2
    features : list ou tuple : la liste des features (ie des dimensions) à représenter
    """
    # Extrait x et y 
    x,y=x_y
    # Taille de l'image (en inches)
    fig, ax = plt.subplots(figsize=(10, 10))
    # Pour chaque composante : 
    for i in range(0, pca.components_.shape[1]):
        # Les flèches
        ax.arrow(0,0, 
                pca.components_[x, i],  
                pca.components_[y, i],  
                head_width=0.07,
                head_length=0.07, 
                width=0.02, )
        # Les labels
        plt.text(pca.components_[x, i] + 0.05,
                pca.components_[y, i] + 0.05,
                features[i])
    # Affichage des lignes horizontales et verticales
    plt.plot([-1, 1], [0, 0], color='grey', ls='--')
    plt.plot([0, 0], [-1, 1], color='grey', ls='--')
    # Nom des axes, avec le pourcentage d'inertie expliqué
    plt.xlabel('F{} ({}%)'.format(x+1, round(100*pca.explained_variance_ratio_[x],1)))
    plt.ylabel('F{} ({}%)'.format(y+1, round(100*pca.explained_variance_ratio_[y],1)))
    # Titre du graphique
    plt.title("Cercle des corrélations (F{} et F{})".format(x+1, y+1), fontsize=16)
    # Le cercle 
    an = np.linspace(0, 2 * np.pi, 100)
    plt.plot(np.cos(an), np.sin(an))  # Add a unit circle for scale
    # Axes et display
    plt.axis('equal')
    plt.show(block=False)

def display_factorial_planes(X_projected, x_y, pca=None, labels = None, clusters=None, alpha=1, figsize=[10,10], marker="." ):
    """
    Affiche la projection des individus

    Positional arguments : 
    -------------------------------------
    X_projected : np.array, pd.DataFrame, list of list : la matrice des points projetés
    x_y : list ou tuple : le couple x,y des plans à afficher, exemple [0,1] pour F1, F2

    Optional arguments : 
    -------------------------------------
    pca : sklearn.decomposition.PCA : un objet PCA qui a été fit, cela nous permettra d'afficher la variance de chaque composante, default = None
    labels : list ou tuple : les labels des individus à projeter, default = None
    clusters : list ou tuple : la liste des clusters auquel appartient chaque individu, default = None
    alpha : float in [0,1] : paramètre de transparence, 0=100% transparent, 1=0% transparent, default = 1
    figsize : list ou tuple : couple width, height qui définit la taille de la figure en inches, default = [10,8] 
    marker : str : le type de marker utilisé pour représenter les individus, points croix etc etc, default = "."
    """
    # Transforme X_projected en np.array
    X_ = np.array(X_projected)
    # On définit la forme de la figure si elle n'a pas été donnée
    if not figsize: 
        figsize = (7,6)
    # On gère les labels
    if  labels is None : 
        labels = []
    try : 
        len(labels)
    except Exception as e : 
        raise e
    # On vérifie la variable axis 
    if not len(x_y) ==2 : 
        raise AttributeError("2 axes sont demandées")   
    if max(x_y )>= X_.shape[1] : 
        raise AttributeError("la variable axis n'est pas bonne")   
    # on définit x et y 
    x, y = x_y
    # Initialisation de la figure       
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    # On vérifie s'il y a des clusters ou non
    c = None if clusters is None else clusters
    # Les points    
    # plt.scatter(   X_[:, x], X_[:, y], alpha=alpha, 
    #                     c=c, cmap="Set1", marker=marker)
    sns.scatterplot(data=None, x=X_[:, x], y=X_[:, y], hue=c,palette="YlGnBu")
    # Si la variable pca a été fournie, on peut calculer le % de variance de chaque axe 
    if pca : 
        v1 = str(round(100*pca.explained_variance_ratio_[x]))  + " %"
        v2 = str(round(100*pca.explained_variance_ratio_[y]))  + " %"
    else : 
        v1=v2= ''
    # Nom des axes, avec le pourcentage d'inertie expliqué
    ax.set_xlabel(f'F{x+1} {v1}')
    ax.set_ylabel(f'F{y+1} {v2}')
    # Valeur x max et y max
    x_max = np.abs(X_[:, x]).max() *1.1
    y_max = np.abs(X_[:, y]).max() *1.1
    # On borne x et y 
    ax.set_xlim(left=-x_max, right=x_max)
    ax.set_ylim(bottom= -y_max, top=y_max)
    # Affichage des lignes horizontales et verticales
    plt.plot([-x_max, x_max], [0, 0], color='grey', alpha=0.8)
    plt.plot([0,0], [-y_max, y_max], color='grey', alpha=0.8)
    # Affichage des labels des points
    if len(labels) : 
        for i,(_x,_y) in enumerate(X_[:,[x,y]]):
            plt.text(_x, _y+0.05, labels[i], fontsize='10', ha='center',va='center') 
    # Titre et display
    plt.title(f"Projection des individus (sur F{x+1} et F{y+1})", fontsize=16)
    plt.show()

def clustering_acp(data, n_clusters=3,n_components=6,x_y=[0,1]):
    """
    Fonction qui permet de:
        Réaliser un clustering via k-means 
        Scaling des données
        ACP avec scree plot
        Cercle des corrélations
        Projections des individus selon les clusters
        Projections des individus selon les clusters avec les centroides
    
    Positional arguments : 
    -----------------------------------
    data : dataframe que l'on souhaite utiliser
    n_clusters : nombre de clusters choisis (voir méthode du coude ou silhouette)
    n_components : choix du nombre de composantes
    x_y : le couple x,y des plans à afficher, exemple [0,1] pour PC1, PC2
    """
    
    # Copie du df_final
    X = data.copy()
    
    #Clustering via k-means pour x clusters
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)
    
    #Enregistrement des labels et centroides des clusters attribués
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    
    # Création d'une colonne "Cluster"
    X["Cluster"] = labels
    
    #On enregistre les clusters dans un df
    X_cluster = X["Cluster"].to_frame()
    
    #On supprime la colonne "Cluster" en prévision de l'ACP
    X.drop(columns="Cluster",inplace=True)
    
    #Enregistrement des colonnes dans une variable
    features = X.columns
    
    #Scaling for ACP
    scaler = StandardScaler()
    X_scaled = X.copy()
    
    #Choix du nombre de composantes
    n_components = n_components
    pca = PCA(n_components=n_components)
    pca.fit(X_scaled)
    
    #Calcul de la variance cumulée
    scree = (pca.explained_variance_ratio_*100).round(2)
    scree_cum = scree.cumsum().round()
    x_list = range(1, n_components+1)

    #Représentation graphique
    plt.figure(figsize=(10,3))
    plt.bar(x_list, scree)
    plt.plot(x_list, scree_cum,c="red",marker='o')
    plt.xlabel("Rang de l'axe d'inertie")
    plt.ylabel("Pourcentage d'inertie")
    plt.title("Eboulis des valeurs propres",fontsize=16, fontweight="bold")
    plt.show(block=False)
    
    print("-"*20)
    print("\n")
    print("Tableau des composantes")
    
    #Calcul des coordonnées des individus dans le nouvel espace
    pcs = pca.components_
    #On met ces valeurs dans un df
    pcs = pd.DataFrame(pcs)
    #On rajoute les colonnes
    pcs.columns = X.columns
    pcs.index = [f"PC{i}" for i in x_list]
    #On transpose pour une meilleure visibilité
    pcs = pcs.T
    display(pcs)
    
    print("-"*20)
    print("\n")
    
    #On enregistre les colonnes dans une variable
    features_acp = pcs.columns
    
    x,y = x_y
    
    fig, ax = plt.subplots(figsize=(10, 10))
    for i in range(0, pca.components_.shape[1]):
        ax.arrow(0,0, pca.components_[x, i], pca.components_[y, i], head_width=0.07, head_length=0.07, width=0.02)
        plt.text(pca.components_[x, i] + 0.05, pca.components_[y, i] + 0.05, X.columns[i])
    
    # affichage des lignes horizontales et verticales
    plt.plot([-1, 1], [0, 0], color='grey', ls='--')
    plt.plot([0, 0], [-1, 1], color='grey', ls='--')


    # nom des axes, avec le pourcentage d'inertie expliqué
    plt.xlabel('PC{} ({}%)'.format(x+1, round(100*pca.explained_variance_ratio_[x],1)))
    plt.ylabel('PC{} ({}%)'.format(y+1, round(100*pca.explained_variance_ratio_[y],1)))

    # titre
    plt.title("Cercle des corrélations (PC{} et PC{})".format(x+1, y+1), fontsize=16, fontweight="bold")

    #ajout du cercle
    plt.plot(np.cos(np.linspace(0, 2 * np.pi, 100)), np.sin(np.linspace(0, 2 * np.pi, 100)), linewidth=2)

    plt.show()
    
    print("-"*20)
    print("\n")
    print("Coordonnées des individus dans le nouvel espace projeté")
    
    X_proj = pca.transform(X_scaled)
    X_proj = pd.DataFrame(X_proj, columns = features_acp)
    display(X_proj)
    
    print("-"*20)
    print("\n")
    
    #Transforme notre df d'origine en array
    X_ = np.array(X_proj)
    
    #On rajoute la colonne "Cluster" à notre df pour afficher la visualisation des clusters
    X = X.merge(X_cluster, how='inner', left_index=True, right_index=True)
    
    plt.figure(figsize=(10,8))

    #Affichage du nuage de points
    sns.scatterplot(x=X_[:, x], y=X_[:, y], hue=X['Cluster'])

    #Nom des axes avec le pourcentage d'inertie
    plt.xlabel('PC{} ({}%)'.format(x+1, round(100*pca.explained_variance_ratio_[x],1)))
    plt.ylabel('PC{} ({}%)'.format(y+1, round(100*pca.explained_variance_ratio_[y],1)))

    # valeur x max et y max
    x_max = np.abs(X_[:, x]).max() *1.1
    y_max = np.abs(X_[:, y]).max() *1.1

    #Affichage des lignes horizontales et verticales
    plt.plot([-x_max, x_max], [0, 0], color='grey', alpha=0.5, ls='--')
    plt.plot([0,0], [-y_max, y_max], color='grey', alpha=0.5, ls='--')

    # on borne x et y 
    plt.xlim(left= -x_max, right=x_max)
    plt.ylim(bottom= -y_max, top=y_max)

    plt.title("Projection des individus (sur PC{} et PC{})".format(x+1, y+1),fontsize=16, fontweight="bold" )

    plt.show()
    
    print("-"*20)
    print("\n")
    print("Tableau des centroides")
    
    centroids_scaled = scaler.fit_transform(centroids)
    centroids_proj = pca.transform(centroids_scaled)
    centroids_proj = pd.DataFrame(centroids_proj, columns=features_acp)
    display(centroids_proj)
    
    print("-"*20)
    print("\n")

    plt.figure(figsize=(10,8))

    #Affichage du nuage de points
    sns.scatterplot(x=X_[:, x], y=X_[:, y], hue=X['Cluster'], alpha=0.7)

    #Affichage des centroides
    markersize = 100
    plt.scatter(centroids_proj.iloc[:, x], centroids_proj.iloc[:, y], markersize,  marker="D", c="orange" )

    #Nom des axes avec le pourcentage d'inertie
    plt.xlabel('PC{} ({}%)'.format(x+1, round(100*pca.explained_variance_ratio_[x],1)))
    plt.ylabel('PC{} ({}%)'.format(y+1, round(100*pca.explained_variance_ratio_[y],1)))

    # valeur x max et y max
    x_max = np.abs(X_[:, x]).max() *1.1
    y_max = np.abs(X_[:, y]).max() *1.1

    #Affichage des lignes horizontales et verticales
    plt.plot([-x_max, x_max], [0, 0], color='grey', alpha=0.5, ls='--')
    plt.plot([0,0], [-y_max, y_max], color='grey', alpha=0.5, ls='--')

    # on borne x et y 
    plt.xlim(left= -x_max, right=x_max)
    plt.ylim(bottom= -y_max, top=y_max)

    plt.title("Projection des individus avec centroides (sur PC{} et PC{})".format(x+1, y+1),fontsize=16, fontweight="bold" )

    plt.show()
    
    print("-"*20)
    print("\n")
    print("DataFrame avec les clusters")
    
    return X

def normal_check(data):
    data_normalized = data.copy()
    
    #Distribution des variables
    plt.figure(figsize=(20,10))

    for i, col in enumerate(data_normalized):
        plt.subplot(2, 5, i+1)
        sns.histplot(data_normalized[col],ec="black", kde=True)
        plt.title(col,fontsize=12)
    plt.show()
    
    #Boxplot pour vérifier les échelles
    data_normalized.boxplot()
    
    #Vérification des variances
    print('Variances')
    print("-"*20)
    print(data_normalized.var().round(2))
    print("\n")
    print("-"*20)
    
    #Test de Kolmogorov-Smirnov

    for i, col in enumerate(data_normalized):
        statistic, p = stats.kstest(data_normalized[col],stats.norm.cdf)
        print(col)
        print("-"*20)
        print("Kolmogorov-Smirnov =",statistic)
        print("P-value =",p)

        if p<0.05:
            print("Rejet de l'hypothèse nulle H0")
        else:
            print("L'hypothèse nulle H0 est acceptée")
    
        print("\n")