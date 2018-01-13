---
layout: post
title:  "Part 5"
date:   2018-01-11
excerpt: "Visualizing Networks of Artists "
image: "/images/Posts_Images/Part5/part5.jpg"
---

In the previous part, we visualized genres' networks.

Using bipartite projections, we will create artists' networks based on their appartenance to common genres: we are taking the problem the other way around.

By doing so, we will reveal an interesting pattern in the data, and create a "recommendation system" based only on the network of artists (we usually refer to recommendation system as machine learning models based on users preferences or similarities).

## Setting up

Let us start as usual, by importing all necessarry modules and data.


```python
%reset

from os import chdir
from pandas import read_csv
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
chdir("C:/Users/antoi/Documents/Spotify_Project")
```

    Once deleted, variables cannot be recovered. Proceed (y/[n])? y
    


```python
artist_df = pd.read_csv("Spotify_Artist_bis.csv", encoding = "ISO-8859-1")
df = pd.DataFrame(artist_df.groupby("Artist")["Genre"].apply(list))
df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Genre</th>
    </tr>
    <tr>
      <th>Artist</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>!!!</th>
      <td>[modern rock, indie pop, indietronica, indie r...</td>
    </tr>
    <tr>
      <th>!Dela Dap</th>
      <td>[nu jazz, electro swing, balkan brass]</td>
    </tr>
    <tr>
      <th>!Distain</th>
      <td>[futurepop, neo-synthpop]</td>
    </tr>
    <tr>
      <th>!PVNDEMIK</th>
      <td>[bass trap, electronic trap]</td>
    </tr>
    <tr>
      <th>!T.O.O.H.!</th>
      <td>[technical death metal]</td>
    </tr>
  </tbody>
</table>
</div>



## Transforming the data
We should first expand the dataframe above to have one observation for each unique genre-artist pair.


```python
edges_df = df.Genre.apply(pd.Series).stack().reset_index(level = 1, drop = True).to_frame('Genre')
edges_df["Artist"] = edges_df.index
edges_df.index.name = None
edges_df = edges_df.reset_index(drop = True)
edges_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Genre</th>
      <th>Artist</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>modern rock</td>
      <td>!!!</td>
    </tr>
    <tr>
      <th>1</th>
      <td>indie pop</td>
      <td>!!!</td>
    </tr>
    <tr>
      <th>2</th>
      <td>indietronica</td>
      <td>!!!</td>
    </tr>
    <tr>
      <th>3</th>
      <td>indie rock</td>
      <td>!!!</td>
    </tr>
    <tr>
      <th>4</th>
      <td>synthpop</td>
      <td>!!!</td>
    </tr>
  </tbody>
</table>
</div>



Next, we will store each artist's popularity in a convenient manner for further use as nodes attributes.


```python
nodes_attr_df = artist_df[["Artist", "Artist Popularity"]].drop_duplicates()
nodes_attr_df["ID"] = nodes_attr_df.index
nodes_attr_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Artist</th>
      <th>Artist Popularity</th>
      <th>ID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Ed Sheeran</td>
      <td>99</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Greyson Chance</td>
      <td>61</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Shane Harper</td>
      <td>52</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Lucy Hale</td>
      <td>53</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Kurt Hugo Schneider</td>
      <td>70</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



Storing it as a dictionnary will also be very convenient afterwards.


```python
nodes_attr_dict = nodes_attr_df.set_index('Artist')["Artist Popularity"].to_dict()
nodes_attr_dict
```




    {'Ed Sheeran': 99,
     'Greyson Chance': 61,
     'Shane Harper': 52,
     'Lucy Hale': 53,
     'Kurt Hugo Schneider': 70,
     'Kina Grannis': 74,
     'Chris Brown': 96,
     "Destiny's Child": 78,
     'G.R.L.': 60,
     'Christopher Wilde': 58,
     'Hearts & Colors': 73,
     'John Legend': 85,
     'will.i.am': 78,
     'Daniel Skye': 58,
     'DJ Snake': 87,
     ...}



## Creating the full graph

We create a graph that will be composed of two kinds of nodes: genre nodes, whose "bipartite" attribute is set to 0, and artist nodes, whose "bipartite" attribute is set to 1. An edge will be created between an artist and a genre according to the first dataframe we presented, using a list comprehension.


```python
B = nx.Graph()
B.add_nodes_from(edges_df.Genre.unique(), bipartite = 0)
B.add_nodes_from(edges_df.Artist.unique() , bipartite = 1)
B.add_edges_from([(row['Genre'], row['Artist']) for idx, row in edges_df.iterrows()])
```

We then create two objects containing the nodes for artists and for genres.


```python
artists = set(n for n, d in B.nodes(data=True) if d['bipartite'] == 1)
genres = set(B) - artists
```

These objects we will be used to draw nodes of separate colors on the following graphs.

Red nodes are genres while blue nodes are artists. However, I do not have enough computing power on my laptop to draw the whole network of artists, so instead I will draw the ego graph from the artist King Krule, with a radius of 2. A radius of 1 would only give the genres of King Krule, and a radius of 2 gives these genres and the artists of these genres.


```python
%matplotlib inline
# del ego
sns.set_style("white", {"axes.facecolor": "1"})

artist = "King Krule"
ego = nx.ego_graph(B, artist, radius = 2, center = False)
ego_artists = set(n for n, d in ego.nodes(data=True) if d['bipartite'] == 1)
ego_genres = set(ego) - ego_artists

for i in list(ego.nodes):
    ego.node[i]["degree"] = ego.degree(i, weight = "weight")

layout = nx.spring_layout(ego)
plt.figure(figsize=(75,75))
nx.draw_networkx_nodes(ego, layout, nodelist = ego_artists, node_color = "blue", alpha = 0.5)
nx.draw_networkx_nodes(ego, layout, nodelist = ego_genres, node_color = "red", alpha = 1, node_size = 5000, with_labels = True)
nx.draw_networkx_edges(ego, layout, alpha = 1)
plt.show()
```


<span class="image fit"><img src="{{ "/images/Posts_Images/Part5/part5.jpg" | absolute_url }}" alt="" /></span>


The nodes we are interested in are the blue nodes whose degree is greater than 1. To this end, we store in the object ``remove`` the nodes whose degree is less than 2. Then, we can create the weighted projected graph of the graph above. This is usefull because edges will have a greater weight when artists share a greater number of genres. We did not use this technique (which is more straightforward) in the previous part because genres had sizes that made comparison meaningless, so the weights had to be rescaled (maybe we could think of an artist size as her number of songs, in a next serie).

Once the graph created, we can remove the nodes of low degree using ``remove``.


```python
remove = [node for node,degree in dict(ego.degree).items() if degree < 2]
ego_G = nx.bipartite.weighted_projected_graph(ego, ego_artists)
ego_G.remove_nodes_from(remove)
```

Next, we compute nodes' betweenness centrality and clustering measures. We then assign it to the nodes' attributes, along with their degree and popularity.


```python
btw = nx.betweenness_centrality(ego_G, weight = "weight")
clustering = nx.clustering(ego_G)

for i in list(ego_G.nodes):
    ego_G.node[i]["popularity"] = nodes_attr_df.loc[nodes_attr_df["Artist"] == i]["Artist Popularity"].values[0]
    ego_G.node[i]["degree"] = ego_G.degree(i, weight = "weight")
    ego_G.node[i]["betweenness"] = btw[i]
    ego_G.node[i]["clustering"] = clustering[i]
```

The next plot shows the result of this weighted projection of the bipartite graph. Nodes are colored according to their degree, and we will be interested in the nodes that have the highest degrees (lighter color). Now every node is an artist, because genre nodes were used for the projection. Additionaly, informations such as degrees or popularities are stored in a convenient manner: they all share the same ordering and will be used for analysis.


```python
%matplotlib inline 
import seaborn as sns
sns.set_style("white", {"axes.facecolor": ".0"})

layout = nx.spring_layout(ego_G, k = 3)
plt.figure(figsize=(100,100))

edges, weights = zip(*{ k:v for k, v in nx.get_edge_attributes(ego_G,'weight').items()}.items())

nodes, degrees = zip(*{ k:v for k, v in nx.get_node_attributes(ego_G,'degree').items()}.items())
nodes, popularities = zip(*{ k:v for k, v in nx.get_node_attributes(ego_G,'popularity').items()}.items())
nodes, betweenness = zip(*{ k:v for k, v in nx.get_node_attributes(ego_G,'betweenness').items()}.items())
nodes, clustering = zip(*{ k:v for k, v in nx.get_node_attributes(ego_G,'clustering').items()}.items())

viz_popularities = [x**2  for x in popularities]
viz_degrees = [x**2 / 100  for x in degrees]

nx.draw_networkx_nodes(ego_G, layout, nodelist = nodes,
                       node_size = viz_degrees,
                       node_color = viz_degrees,
                       cmap = plt.cm.viridis, alpha = 0.5)
nx.draw_networkx_edges(ego_G, layout, edgelist = edges, edge_color = weights, alpha = 0.1, edge_cmap = plt.cm.Reds)

plt.show()
```


<span class="image fit"><img src="{{ "/images/Posts_Images/Part5/part5-2.jpg" | absolute_url }}" alt="" /></span>


We use nodes informations computed from this graph (betweenness, clustering and degree) and others that were carried during all of the transformations (name and popularity), and store them in a dataframe.


```python
import numpy as np

degree_df = pd.DataFrame({"Node":nodes, "Degree":degrees, "Betweenness":betweenness, "Popularity":popularities,
                          "Clustering": clustering}
                        ).sort_values("Popularity", ascending = True)
degree_df.head(10)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Betweenness</th>
      <th>Clustering</th>
      <th>Degree</th>
      <th>Node</th>
      <th>Popularity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>535</th>
      <td>0.000282</td>
      <td>0.980132</td>
      <td>618</td>
      <td>Starfucker</td>
      <td>7</td>
    </tr>
    <tr>
      <th>557</th>
      <td>0.000282</td>
      <td>0.980132</td>
      <td>618</td>
      <td>Violens</td>
      <td>21</td>
    </tr>
    <tr>
      <th>272</th>
      <td>0.000282</td>
      <td>0.980132</td>
      <td>618</td>
      <td>Ford &amp; Lopatin</td>
      <td>22</td>
    </tr>
    <tr>
      <th>77</th>
      <td>0.000282</td>
      <td>0.980132</td>
      <td>618</td>
      <td>Suckers</td>
      <td>24</td>
    </tr>
    <tr>
      <th>79</th>
      <td>0.000282</td>
      <td>0.980132</td>
      <td>618</td>
      <td>Sun Airway</td>
      <td>28</td>
    </tr>
    <tr>
      <th>369</th>
      <td>0.000282</td>
      <td>0.980132</td>
      <td>618</td>
      <td>Deastro</td>
      <td>28</td>
    </tr>
    <tr>
      <th>263</th>
      <td>0.000282</td>
      <td>0.980132</td>
      <td>618</td>
      <td>Air France</td>
      <td>30</td>
    </tr>
    <tr>
      <th>454</th>
      <td>0.000147</td>
      <td>0.973159</td>
      <td>655</td>
      <td>Enon</td>
      <td>31</td>
    </tr>
    <tr>
      <th>577</th>
      <td>0.000147</td>
      <td>0.973159</td>
      <td>655</td>
      <td>Quasi</td>
      <td>31</td>
    </tr>
    <tr>
      <th>444</th>
      <td>0.000282</td>
      <td>0.980132</td>
      <td>618</td>
      <td>Young Magic</td>
      <td>31</td>
    </tr>
  </tbody>
</table>
</div>



## Creating a function plotting nodes information for any artist
The function defined below does exactly the same as what we did before to obtain a similar dataframe as above starting with the artists King Krule for the first ego graph, but for any artist, without the plots for the graphs and, at the end, with code to plot nodes information from the dataframe.


```python
def pop_graph(artist):
    
    # Obtain the ego graph from the desired artists
    ego = nx.ego_graph(B, artist, radius = 2, center = False)
    
    # Obtain separate lists of nodes from this new graph for genres and artists
    ego_artists = set(n for n, d in ego.nodes(data=True) if d['bipartite'] == 1)
    ego_genres = set(ego) - ego_artists

    # Set "degree" attributes of nodes to their degree, i.e. how many artists they share genres with
    for i in list(ego.nodes):
        ego.node[i]["degree"] = ego.degree(i, weight = "weight")
        
    # Get the nodes with a degree of 1, which will be removed
    remove = [node for node,degree in dict(ego.degree).items() if degree < 2]
    
    # Obtain a new graph, ego_G, which is the weighted projection of the previous ego graph, which was bipartite. 
    # The projection is made such that an edge is drawn between artists that share at least a genre, the weights  of these
    # edges will be the number of shared genres
    ego_G = nx.bipartite.weighted_projected_graph(ego, ego_artists)
    
    # Remove the nodes now
    ego_G.remove_nodes_from(remove)
    
    # For this new graph, set the attributes of popularity, degree, betweenness and clustering to its nodes
    btw = nx.betweenness_centrality(ego_G, weight = "weight")
    clustering = nx.clustering(ego_G)
    for i in list(ego_G.nodes):
        ego_G.node[i]["popularity"] = nodes_attr_df.loc[nodes_attr_df["Artist"] == i]["Artist Popularity"].values[0]
        ego_G.node[i]["degree"] = ego_G.degree(i, weight = "weight")
        ego_G.node[i]["betweenness"] = btw[i]
        ego_G.node[i]["clustering"] = clustering[i]
    
    # Retrieve these values with their respective nodes. This ensures that each lists will be ordered the same way
    edges, weights = zip(*{ k:v for k, v in nx.get_edge_attributes(ego_G,'weight').items()}.items())
    nodes, degrees = zip(*{ k:v for k, v in nx.get_node_attributes(ego_G,'degree').items()}.items())
    nodes, popularities = zip(*{ k:v for k, v in nx.get_node_attributes(ego_G,'popularity').items()}.items())
    nodes, betweenness = zip(*{ k:v for k, v in nx.get_node_attributes(ego_G,'betweenness').items()}.items())
    nodes, clustering = zip(*{ k:v for k, v in nx.get_node_attributes(ego_G,'clustering').items()}.items())
    
    # Define a dataframe from these values, which will be used next to draw the graph
    degree_df = pd.DataFrame({"Node":nodes, "Degree":degrees, "Betweenness":betweenness, "Popularity":popularities,
                              "Clustering": clustering}
                            ).sort_values("Popularity", ascending = False)
    
    # Draw the graph
    sns.set_style("white")
    
    f, (ax1, ax2, ax3) = plt.subplots(nrows = 1, ncols = 3, figsize = (20, 10), sharey = True)
    
    ax1.set_ylim(-1, 101)
    ax2.set_xlim(-0.001,
                 (max(degree_df.Betweenness) + 0.05 * max(degree_df.Betweenness)))
    ax3.set_xlim((min(degree_df.Clustering) - 0.05 * min(degree_df.Clustering)),
                 (max(degree_df.Clustering) + 0.05 * max(degree_df.Clustering)))
    
    ax2.yaxis.label.set_visible(False)
    ax3.yaxis.label.set_visible(False)
    
    sns.regplot(x = "Degree", y = "Popularity", data = degree_df, order = 1, ax = ax1)
    sns.regplot(x = "Betweenness", y = "Popularity", data = degree_df, order = 1, ax = ax2)
    sns.regplot(x = "Clustering", y = "Popularity", data = degree_df, order = 1, ax = ax3)
    
    sns.despine()
        
    plt.show()
```

## Using the function for analysis
Let us call our newly created function on different artists.


```python
pop_graph("808 State")
```


<span class="image fit"><img src="{{ "/images/Posts_Images/Part5/part5-3.png" | absolute_url }}" alt="" /></span>




```python
pop_graph("St Germain")
```


<span class="image fit"><img src="{{ "/images/Posts_Images/Part5/part5-4.png" | absolute_url }}" alt="" /></span>




```python
pop_graph("BadBadNotGood")
```


<span class="image fit"><img src="{{ "/images/Posts_Images/Part5/part5-5.png" | absolute_url }}" alt="" /></span>




```python
pop_graph("Kink")
```


<span class="image fit"><img src="{{ "/images/Posts_Images/Part5/part5-6.png" | absolute_url }}" alt="" /></span>



```python
pop_graph("Mild High Club")
```


<span class="image fit"><img src="{{ "/images/Posts_Images/Part5/part5-7.png" | absolute_url }}" alt="" /></span>



```python
pop_graph("King Krule")
```


<span class="image fit"><img src="{{ "/images/Posts_Images/Part5/part5-8.png" | absolute_url }}" alt="" /></span>



```python
pop_graph("Froth")
```


<span class="image fit"><img src="{{ "/images/Posts_Images/Part5/part5-9.png" | absolute_url }}" alt="" /></span>


Because these are computationaly intensive, I cannot call the function on hundreds of artists. However, on the few I did, there is already a really interesting pattern: 
* **An artist's popularity seems to always be positively correlated with its degree**;
* **An artist's popularity seems to always be negatively correlated with its clustering coefficient**.

There does not seem to be, however, any pattern regarding popularity and betweennes.

Of course, the claims above should be taken with great care: they result from the observation of a few samples (but among all of those that I ran and are not shown here to not take too much space, none followed a different pattern). It would be very interesting to run the function on a large number of randomly selected artists, storing the slopes of the regression lines and observing their distribution.

## Is the network of artists scale-free ?

Let us create the weighted projected graph from the whole bipartite graph. This process is very computationaly intensive and takes a lot of time on my laptop.


```python
G = nx.bipartite.weighted_projected_graph(B, artists)
```

We can then do as we did and explained in the previous Part, and plot the distribution of degrees on a loglog scale.


```python
deg_hist = nx.degree_histogram(G)
x = range(len(deg_hist))
y = deg_hist
df = pd.DataFrame({"Degree": x, "Frequency": y})
df["Frequency"] = df["Frequency"] / sum(df["Frequency"])
df = df[df.Frequency > 0]
df = df[df.Degree > 0]
df["log_Degree"] = np.log(df.Degree)
df["log_Frequency"] = np.log(df.Frequency)
df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Degree</th>
      <th>Frequency</th>
      <th>log_Degree</th>
      <th>log_Frequency</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.000191</td>
      <td>0.000000</td>
      <td>-8.563271</td>
    </tr>
    <tr>
      <th>12</th>
      <td>12</td>
      <td>0.000021</td>
      <td>2.484907</td>
      <td>-10.760495</td>
    </tr>
    <tr>
      <th>18</th>
      <td>18</td>
      <td>0.000037</td>
      <td>2.890372</td>
      <td>-10.200879</td>
    </tr>
    <tr>
      <th>33</th>
      <td>33</td>
      <td>0.000042</td>
      <td>3.496508</td>
      <td>-10.067348</td>
    </tr>
    <tr>
      <th>36</th>
      <td>36</td>
      <td>0.000186</td>
      <td>3.583519</td>
      <td>-8.591442</td>
    </tr>
  </tbody>
</table>
</div>



We have some outliers (that were found through plotting) and that will be removed. They represent 3.2% of the points in the dataframe above.


```python
print("Removing outliers: " + str(100 * (sum(df.log_Degree <= 5) / len(df.log_Degree))) + "% of data")
```

    Removing outliers: 3.21070234114% of data
    


```python
import seaborn as sns
%matplotlib inline
sns.set_style("white", {"axes.facecolor": "1"})
sns.lmplot("log_Degree", "log_Frequency", data = df[df.log_Degree > 5], aspect = 2, size = 8)
plt.show()
```


<span class="image fit"><img src="{{ "/images/Posts_Images/Part5/part5-10.png" | absolute_url }}" alt="" /></span>



```python
import statsmodels.formula.api as smf
results = smf.ols('log_Frequency ~ log_Degree', data = df[df.log_Degree > 5]).fit()
print(results.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:          log_Frequency   R-squared:                       0.908
    Model:                            OLS   Adj. R-squared:                  0.908
    Method:                 Least Squares   F-statistic:                 2.854e+04
    Date:                Sun, 07 Jan 2018   Prob (F-statistic):               0.00
    Time:                        19:31:31   Log-Likelihood:                -2348.8
    No. Observations:                2894   AIC:                             4702.
    Df Residuals:                    2892   BIC:                             4713.
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept      6.3883      0.098     65.283      0.000       6.196       6.580
    log_Degree    -2.2817      0.014   -168.952      0.000      -2.308      -2.255
    ==============================================================================
    Omnibus:                      106.547   Durbin-Watson:                   1.856
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              126.262
    Skew:                          -0.431   Prob(JB):                     3.82e-28
    Kurtosis:                       3.553   Cond. No.                         71.3
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    

Looking at the plot that shows a linear degree distribution on the loglog scale, and the regression summary above, it appears that the network of artists is described by the power law:
$$ P(k) \sim k^{-2.3}$$
Which is very common in scale-free networks (recal from the previous part that it usually ranges between -3 and -2.

**It is thus very likely that the network of artists is scale-free and is formed by preferential attachment. It contains artists that act as "hubs" and create a "small-world" of artists.**

To confirm that it is scale-free, it would be interesting to compute other metrics, such as the average shortest distance between any two nodes but, again and unfortunately, I don't have enough computing power on my laptop with such a large network.

## Recommanding artists based on genres only

We will now build a function that suggests other artists to listen based on a given artist. This is certainly everything but new, but it is still fun to do it yourself. It is quite similar to what we did already, except this time we will only focus on the degrees of the nodes and return those with the highest degrees. These nodes are those that share the most genres with the given artist. This criterion is extremely simple but can prove to be very powerful. 


```python
def network_recommendation(artist):
    
    """"
    Given an artist's name, return in descending order a list of suggestions of artists to listen to based on genre proximity
    """"
    
    ### THIS PART IS THE SAME AS BEFORE ###
    ego = nx.ego_graph(B, artist, radius = 2, center = False)
    
    ego_artists = set(n for n, d in ego.nodes(data=True) if d['bipartite'] == 1)
    ego_genres = set(ego) - ego_artists
    
    remove = [node for node,degree in dict(ego.degree).items() if degree < 2]
    ego_G = nx.bipartite.weighted_projected_graph(ego, ego_artists)
    ego_G.remove_nodes_from(remove)
    
    ### STORE NODES AND THEIR DEGREES ###
    nodes = list(dict(nx.degree(ego_G, weight = "weight")).keys())
    degree = list(dict(nx.degree(ego_G, weight = "weight")).values())
    degree_df = pd.DataFrame({"Node":nodes, "Degree":degree}).sort_values("Degree", ascending = False)
    
    ### ADD NODES WITH HIGHEST DEGREES IN DESCENDING ORDER, STOP WHEN THERE IS AT LEAST 5 RECOMMENDATIONS,
    ### THE DISPLAY THE RANK OF THE RECOMMENDATION, IF TWO ARTISTS HAVE THE SAME DEGREE THEY WILL HAVE THE SAME RANK
    recommendations = []
    i = 1
    while len(recommendations) < 6:
        recommend2 = degree_df[degree_df["Degree"] == sorted(set(degree))[-i]]
        recommendations2 = list(recommend2["Node"])
        for reco in recommendations2:
            recommendations.append(reco + ", rank: " + str(i))
        i += 1
    
    return recommendations
```

Let us have a look at our suggestions. 


```python
network_recommendation("Kamasi Washington")
```




    ['Christian Scott aTunde Adjuah, rank: 1',
     'Pharoah Sanders, rank: 1',
     'Yusef Lateef, rank: 1',
     'Ahmad Jamal, rank: 1',
     'Brad Mehldau, rank: 1',
     'Ahmad Jamal Trio, rank: 1',
     'Joshua Redman, rank: 1']




```python
network_recommendation("Thee Oh Sees")
```




    ['Parquet Courts, rank: 1',
     'Ty Segall, rank: 1',
     'Deerhunter, rank: 2',
     'Guided By Voices, rank: 3',
     'Cloud Nothings, rank: 4',
     'DIIV, rank: 5']




```python
network_recommendation("Kerri Chandler")
```




    ['Inner City, rank: 1',
     'Floorplan, rank: 1',
     'Terrence Parker, rank: 2',
     'Moodymann, rank: 3',
     'Robert Hood, rank: 3',
     'Ron Trent, rank: 3',
     'Palms Trax, rank: 3']



### Conclusion on the recommendations
From my point of view, these recommendations are surprisingly good! I find a lot of artists I would have recommended myself, based on an artist and, most importantly, I was introduced to new artists I wish I had discovered earlier (I actually used it a lot for myself since I wrote it)!

# Summary

In this part:
* We used weighted bipartite projections to build networks of artists, based on their common genres;
* We found that an artist's Popularity might be positively linked with its Degree, and negatively with its Clustering.
* We also found that the network of musical artists most likely is scale-free and a small-world;
* Finally, we programmed a small yet effective recommendation function.

In the next section, we will try to predict an artist popularity, knowing only its genres, using Machine Learning techniques (Gradient Boosting and Neural Networks).
