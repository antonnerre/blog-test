---
layout: post
title:  "Part 4"
date:   2018-01-11
excerpt: "Visualizing Networks of Genres "
image: "./images/Posts_Images/Part4/part4_resized.jpg"
---

In this part, we will use the package ``networkx`` to analyze and visualize networks of musical genres, based on the dataset we created in the previous part.

We will reveal that a few genres are very "central" in the full network, and that this network seems to be scale-free, meaning it might follow preferential attachment: there might be a Matthew effect on genres' influences, that is, more and more genre form around the already popular genres, and less and less around the least popular genres.

## Setting up

Let us import all modules...


```python
%reset

import networkx as nx
import matplotlib.pyplot as plt
from os import chdir
from pandas import read_csv
from scipy.sparse import csr_matrix
from numpy import log
from numpy import array
import pandas as pd
import numpy as np
import seaborn as sns

chdir("C:/Users/antoi/Documents/Spotify_Project")
```

    Once deleted, variables cannot be recovered. Proceed (y/[n])? y
    

... and load the necessary datasets (which were all created in previous Parts).

Some genres have a very low number of common artists. We will delete edges between genres that share less than 1% of common artists, as it will make our computations much faster and because these edges are virtually non-existent.


```python
genres = read_csv("Ordered_Genres.csv")
node_sizes = list(genres["Size"])
node_popularities = list(genres["Popularity"])
labels = dict(enumerate(genres["Genre"], start = 0))

artists = pd.read_csv("Spotify_Artist_bis.csv", encoding = "ISO-8859-1")
artists = pd.DataFrame(artists.groupby("Artist")["Genre"].apply(list))

adj_df = read_csv("Adjacency_Matrix.csv", index_col = 0)
adj_df[adj_df <= 1] = 0
adj_df.head()
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
      <th>dance pop</th>
      <th>modern rock</th>
      <th>pop</th>
      <th>latin</th>
      <th>...</th>
      <th>football</th>
      <th>anime cv</th>
      <th>fussball</th>
      <th>kayokyoku</th>
    </tr>
    <tr>
      <th>Genre</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>dance pop</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>35.305344</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>modern rock</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.654743</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>pop</th>
      <td>35.305344</td>
      <td>3.654743</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>latin</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>pop rap</th>
      <td>12.835093</td>
      <td>0.000000</td>
      <td>10.230179</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



We can start by visualizing the genres network. The adjacency matrix is used to create edges between genres that have common artists. The weights given to these edges corresponds to the values in the matrix: the greater the share of common artists, the greater the weight. The graph below shows the following:
* Nodes: the larger and brighter the node, the larger the size of the genre;
* Edges: the darker (more red) the edge, the larger the weight.


```python
%matplotlib inline

# transform the data to a sparse matrix for faster computations
graph = csr_matrix(adj_df.values)

# use the log of weights to normalize the data (makes visualizations clearer)
graph.data = log(graph.data)

# initiate the graph
G = nx.Graph(graph)

# write nodes' attributes
for n in range(len(labels)):
    G.nodes[n]["label"] = list(labels.values())[n]
    
for n in range(len(node_sizes)):
    G.nodes[n]["size"] = node_sizes[n]
    
for n in range(len(node_sizes)):
    G.nodes[n]["popularity"] = node_popularities[n]
    
for n in range(len(labels)):
    G.nodes[n]["degree"] = list(dict(G.degree()).values())[n]

# define the layout for drawing
layout = nx.spring_layout(G, k = 1)

# recover nodes attributes in a convenient format for plotting
pos = [[item[0], item[1]] for item in layout.values()]
edges, w = zip(*nx.get_edge_attributes(G,'weight').items())
nodes, l = zip(*nx.get_node_attributes(G,'label').items())
nodes, s = zip(*nx.get_node_attributes(G,'size').items())
nodes, p = zip(*nx.get_node_attributes(G,'popularity').items())

# increase differences for more distinction with the color palettes and 
p = [x ** 2 for x in p]
w = [x ** 4 for x in w]

# increase the size of nodes 
s = [x * 10 for x in s]

# store some attributes conveniently for networkx
l = dict(zip(nodes, l))
d = list(dict(G.degree()).values())

# define figure size
plt.figure(figsize=(100,100))

# actually draw nodes and edges
nx.draw_networkx_nodes(G, pos, node_size = s, node_color = p, cmap = plt.cm.viridis, alpha = 0.7)
nx.draw_networkx_edges(G, pos, edge_color = w, edge_cmap = plt.cm.autumn, alpha = 0.2)
plt.show()
```


<span class="image fit"><img src="{{ "/images/Posts_Images/Part4/part4_resized.jpg" | absolute_url }}" alt="" /></span>

What can be seen here ? 
* First of all, this visualization reveals that **the largest genres are "central" (they have the highest degrees, or number of connexions) to the network**: nodes tend to be larger in the middle of the graph, and smaller at the periphery;
* Secondly, it appears that **the most popular genres are also the most central**: nodes tend to be of brighter color in the middle of the graph, and darker at the periphery;
* Thirdly, **the largest shares of common artists are concentrated between these central genres**: notice how the edges are darker in the area in the middle where there are the central nodes.

It thus seems that the network of genres is dominated by a few very **large** (*size of nodes*) and **popular** (*color of nodes*) genres that have **a lot** (*"centrality" of the nodes on the graph*) of **strong connexions** (*color of edges*) between each other, forming a core. Other genres lay around this core.

Let us try now to get a grasp of how this network might have formed.
## Is it random ? 
Following the Erdős–Rényi model, if this network was formed following a random model, then links would be created with probability p between two genres, with $$p$$ such that: 

$$d = np \Leftrightarrow p = \frac{d}{n}$$

where $$d$$ is the average degree in the graph and $$n$$ is the number of nodes. Let us compute $$p$$.


```python
deg_hist = nx.degree_histogram(G)
deg_hist_list = np.repeat(range(len(deg_hist)), deg_hist, axis=0)
d = np.mean(deg_hist_list)
n = len(deg_hist_list)
p = d / n
p
```




    0.010490824099722991



If it actually follows a random model, then we can compare this $$p$$ to two particular thresholds:

$$t_1 = \frac{1}{n}$$

and 

$$t_2 = \frac{\log(n)}{n} $$

* If $$p > t_1$$, the network should have a unique giant component;
* If $$p > t_2$$, the network should be fully connected.

Let us see what properties our network should verify were it random, by computing these thresholds and comparing them with $$p$$.


```python
thresh1 = 1 / n
thresh2 = np.log(n) / n
print("If the network comes from an Erdos Renyi model:")
if (p > thresh1):
    print("   The network should have a unique giant component.")
if (p > thresh2):
    print("   The network should be fully connected.")
```

    If the network comes from an Erdos Renyi model:
       The network should have a unique giant component.
       The network should be fully connected.
    

Let us check if our network verifies both properties.

### Giant component &  connectivity
We start by computing all connected component subgraphs (if there are any) from our network, and print each of these graphs' orders, from highest to lowest.


```python
connected = sorted(nx.connected_component_subgraphs(G, copy = True), key = len, reverse=True)
orders = []
for i in range(len(connected)):
    orders.append(connected[i].order())
print(np.array(orders))
print("Number of connected components: " + str(len(orders)))
```

    [1372   12    7    6    4    3    3    3    3    2    2    2    2    2    2
        2    2    2    2    2    1    1    1    1    1    1    1    1    1    1
        1    1    1    1    1    1    1    1    1    1    1    1    1    1    1
        1    1    1    1    1    1    1    1    1    1    1    1    1    1    1
        1    1    1    1    1    1    1    1    1    1    1    1    1    1    1
        1    1    1    1    1    1    1    1    1    1    1    1    1    1    1
        1    1    1    1    1    1    1    1    1    1    1    1    1    1    1]
    Number of connected components: 105
    

It is clear that this network is not fully connected as it contains 105 connected components. It does, however, contain a giant component: the first connected component has an order of 1372, and no other has an order above 12.

**We can thus conclude that the network of genres is most probably not random.**

## Cultural proximity in the connected components

Another way to support the previous statement is found below.

It is really interesting to notice what genres compose these small, connected networks, which have no link to the giant component. Even if these genres might have a link with genre in the giant component (strictly musically, "dutch pop" could be related to "pop"), the creation and naming of the genres reveals a lot about other aspects of music: more than a similarity in the sound, music is also a way to create identities to which both artists and listeners can associate. 


```python
for i in list(range(0, 21)):
    print("Component " + str(i+1) + ": " + str(genres.loc[list(connected[i+1].nodes),:].Genre.tolist()))
```

    Component 1: ['deep dutch hip hop', 'dutch pop', 'levenslied', 'dutch hip hop', 'belgian rock', 'classic dutch pop', 'classic belgian pop', 'carnaval', 'dutch rock', 'indorock', 'belgian indie', "traditional rock 'n roll"]
    Component 2: ['polish pop', 'polish hip hop', 'polish punk', 'classic polish pop', 'disco polo', 'polish reggae', 'polish indie']
    Component 3: ['czech folk', 'czech rock', 'classic czech pop', 'slovak pop', 'slovak hip hop', 'czech hip hop']
    Component 4: ['turbo folk', 'yugoslav rock', 'croatian pop', 'klapa']
    Component 5: ['fake', 'covertrance', 'workout']
    Component 6: ['nursery', "children's music", 'kids dance party']
    Component 7: ['romanian pop', 'manele', 'romanian rock']
    Component 8: ['a cappella', 'college a cappella', 'barbershop']
    Component 9: ['latin christian', 'deep latin christian']
    Component 10: ['persian pop', 'persian traditional']
    Component 11: ['afrikaans', 'african rock']
    Component 12: ['galego', 'spanish folk']
    Component 13: ['chinese traditional', 'japanese traditional']
    Component 14: ['sega', 'malagasy folk']
    Component 15: ['bulgarian rock', 'chalga']
    Component 16: ['c64', 'demoscene']
    Component 17: ['swiss hip hop', 'swiss rock']
    Component 18: ['chinese indie rock', 'chinese experimental']
    Component 19: ['guidance', 'motivation']
    Component 20: ['brazilian ccm']
    Component 21: ['barnmusik']
    

Looking at the result above, it seems that many of the connected components are based on cultural proximity (common language or history, geographical proximity, etc.) which are very important and might create networks of artists disconnected from the giant components. Here this proximity is more defining than the musical genre itself: "dutch pop", "polish hip hop", "romanian pop", "swiss rock", etc. are dutch, polish, romanian and swiss before anything else.

* For instance, in the first component above, there is "dutch pop", and "belgian rock", and this is certainly due to the fact that Dutch is spoken in the Netherlands and in the Flemish part of Belgium. Moreover, "levenslied" means "life song" in Dutch, "indorock" originated in the Netherlands thanks to migrants coming from Indonesia after its independence from the Netherlands in 1945.
* In the second component we find genres sung in Polish.
* In the third component there are genres from the former state Czecho-Slovakia, dissolved in 1993, such as "czech folk" and "slovak pop".
* In the fourth component we can find "turbo folk", a genre originating from Serbia, "klapa", a genre originating from Croatia, as well as "croatian pop" and "yugoslav rock". Serbia and Croatia both have a long and complicated history with the former state of Yugoslavia (see this if interested: https://en.wikipedia.org/wiki/Breakup_of_Yugoslavia).
* In the fifth component, I honestly have no idea of what those genres are supposed to encompass so I will not comment.
* In the sixth component, we find children music.
* In the seventh, we have music from Romania, "romanian pop" and "romanian rock", as well as "manele", a music genre created by the Romani people in Romania. 

An so on and so forth, from sino-japanese traditional music to music from Madagascar and Mauritius, or music from Bulgaria, Switzerland, etc. 

### Does it come from preferential attachment ?

This is suggested from the previous results and, as new music often arises from the collaboration of different persons that share some cultural influences, it is quite plausible to expect our network to be formed by preferential attachment. These networks are characterized by degree distributions following power laws. In this case, the distribution should be linear on a loglog scale.

Let us have a look at the degree distrubution for the network.


```python
deg_hist_giant = nx.degree_histogram(G)
x = range(len(deg_hist_giant))
y = deg_hist_giant
df = pd.DataFrame({"Degree": x, "Frequency": y})
df["Frequency"] = df["Frequency"] / sum(df["Frequency"])
df = df[df.Frequency > 0]
df = df[df.Degree > 1]
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
      <th>2</th>
      <td>2</td>
      <td>0.090789</td>
      <td>0.693147</td>
      <td>-2.399212</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0.091447</td>
      <td>1.098612</td>
      <td>-2.391992</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0.056579</td>
      <td>1.386294</td>
      <td>-2.872118</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>0.042763</td>
      <td>1.609438</td>
      <td>-3.152078</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>0.042105</td>
      <td>1.791759</td>
      <td>-3.167583</td>
    </tr>
  </tbody>
</table>
</div>



As can be seen below, the relationship indeed looks linear. 


```python
import seaborn as sns
%matplotlib inline
sns.lmplot("log_Degree", "log_Frequency", data = df, fit_reg = False)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x247f61800f0>




<span class="image fit"><img src="{{ "/images/Posts_Images/Part4/part4-2.png" | absolute_url }}" alt="" /></span>

As explained earlier, scale-free networks have their degree distribution described by power laws: 

$$ P(k) \sim k^{-\gamma} $$

Lets us have a look at the coefficient $$\gamma$$. It corresponds to the negative of the estimated coefficient for $$\log(k)$$ in a Least Squares regression such that:

$$ P = a \times k^{-\gamma}$$

$$ \Leftrightarrow \log(P) = \alpha + \beta \times \log(k)$$

So $$\gamma = - \beta$$. 


```python
import statsmodels.formula.api as smf
results = smf.ols('log_Frequency ~ log_Degree', data = df).fit()
print(results.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:          log_Frequency   R-squared:                       0.875
    Model:                            OLS   Adj. R-squared:                  0.873
    Method:                 Least Squares   F-statistic:                     670.7
    Date:                Sun, 07 Jan 2018   Prob (F-statistic):           4.21e-45
    Time:                        15:11:56   Log-Likelihood:                -61.900
    No. Observations:                  98   AIC:                             127.8
    Df Residuals:                      96   BIC:                             133.0
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept     -0.6000      0.201     -2.992      0.004      -0.998      -0.202
    log_Degree    -1.3673      0.053    -25.899      0.000      -1.472      -1.262
    ==============================================================================
    Omnibus:                        4.241   Durbin-Watson:                   1.974
    Prob(Omnibus):                  0.120   Jarque-Bera (JB):                3.801
    Skew:                          -0.478   Prob(JB):                        0.149
    Kurtosis:                       3.124   Cond. No.                         17.5
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    

We obtain the following: 

$$\gamma = 1.37$$

This means that our network for genres is described by the following power law:

$$P(k) \sim k^{-1.37}$$

The most common values for $$\gamma$$ ranges from 2 to 3. Indeed, the Barabasi-Albert model for instance has $$\gamma = 3$$, so we can reject this model. The non-linear preferential attachment is not plausible as well because it has $$\gamma \geq 2$$.

There is not actually any easily tractable model for $$\gamma < 2$$, but if you are interested you can read the following: 

https://www.researchgate.net/publication/7069628_Scale-free_networks_with_an_exponent_less_than_two

In particular, it explains that, in these models where the exponent is less than two, the "number of links grows faster than the number of nodes and they naturally posses the small world property".

**We can thus conclude that the network of genres most probably is scale-free and a small world, which contains genres that act as "hubs", just like we saw in the first plot. As it is scale-free, it most likely emerges from some sort of preferential attachment.**

# Bonus: producing a function mapping a given artist's musical identity

We will program a function that plots the graph of the sub-network defined by a given artist. The function will look for the artist in the dataset, find its genres and the genres related to the artist's genre, to create a sort of map of influencing genres of the artist. 

If artists belonged to a single genre, this would be an ego graph, which can be done simply using ``networkx`` (without edges between non-home nodes). The difficulty of the task here comes from the fact that artists can belong to several genres at the same time. Moreover, we will be using particular coloring that will recquire network manipulations.

Let us start with this short function that will be usefull in the main one.


```python
def find_key(input_dict, value):  
    """
        Given a dictionnary and a value to look for, returns the corresponding key in the dictionnary.
    """
    
    for k, v in input_dict.items():
        if v == value:
            yield k
```

Then the main function, whose steps are detailled below.


```python
def search(search_list, artist, x, y, lw, k, lab):
    """
        This function looks for the artist's genres in the dataset, get the related genres and their relevant data,
        then plots these 'combined ego-graphs'.
        
        search_list: the list of genres to use as home nodes
        artist: the artist name if given
        x: width of the plot
        y: height of the plot
        lw: linewidth for edges
        k: distance parameter for the spring layout
        lab: set to True if want labels on the plot
    """
    
    # initiate empty lists for the ego-graphs and their home node
    graph_list = list()
    nodes_homes_list = list()

    # this for loop screens the dataset using the find_key() function defined above
    for i in range(len(search_list)):
        
            # get the keys for the genre i in the list of genres passed to the function
            n = list(find_key(labels, str(search_list[i])))[0]
            
            # add it to the list of home nodes
            nodes_homes_list.append(n)
            
            # create the ego-graph for the genre i in the list of genres
            sub_G = nx.ego_graph(G, n) 
            
            # store its weight data
            sub_edges, sub_weights = zip(*nx.get_edge_attributes(sub_G,'weight').items())
            sub_edge_weights = dict(zip(sub_edges, sub_weights))
            sub_list = list(sub_edge_weights.keys())
            
            # remove edges between nodes that are both not homes from the ego-graph, this is for clarity of the visualization
            remove = list()
            for i in range(len(sub_list)):
                if n not in sub_list[i] :
                    remove.append(sub_list[i])
            sub_G.remove_edges_from(remove)
            
            # append the newly created ego-graph to the list of graphs that will be used to compose the final one
            graph_list.append(nx.ego_graph(sub_G, n, distance = "weight"))

    # start with the first graph in the list      
    V = graph_list[0]

    # this for loops adds the individual ego-graphs one after the other the final, large one.
    for j in range(1, len(search_list)):
        
        # compose larger ego-graph from the previous one, adding the one from genre j in the list of genre provided
        V = nx.compose(V , graph_list[(j)])
    
    
    
    ##### PLOTTING STARTS HERE #####
    
    
    
    # define the base layout
    layout = nx.spring_layout(V, k = k, iterations = 1000)
    
    # store data one node positions, node data, edge data etc.
    pos = [[item[0], item[1]] for item in layout.values()]
    edges, w = zip(*nx.get_edge_attributes(V,'weight').items())
    nodes, l = zip(*nx.get_node_attributes(V,'label').items())
    nodes, s = zip(*nx.get_node_attributes(V,'size').items())
    nodes, p = zip(*nx.get_node_attributes(V,'popularity').items())
    
    # increase the differences for easier visualization
    p = [x ** 2 for x in p]
    s = [x ** 1.5 for x in s]
    w = [x ** 2 for x in w]
    
    # prepare the right format for plotting
    p_dict = dict(zip(nodes,p))
    l = dict(zip(nodes, l))
    d = list(dict(V.degree()).values())
    
    # separate home nodes (genres in the provided list) from the others (related genres)
    nodes_others_list = [x for x in nodes if x not in nodes_homes_list]
    nodes_others_colors = [p_dict[x] for x in nodes_others_list]
    labels_others = {k: l[k] for k in tuple(nodes_others_list)}
    labels_homes = {k: l[k] for k in tuple(nodes_homes_list)}
    
    # set some parameters for the plot
    plt.rcParams["font.family"] = "serif"
    plt.figure(figsize=(x,y))
    cmap = plt.cm.bwr
    edge_cmap = plt.cm.bwr
    
    # draw the home nodes in black, with size depending on genre size
    nx.draw_networkx_nodes(V, layout, nodelist = nodes_homes_list, node_size = s, node_color = "black")
    
    # draw the other nodes, with color depending on popularity, and size on genre size
    nx.draw_networkx_nodes(V, layout, nodelist = nodes_others_list, node_size = s, node_color = nodes_others_colors, cmap = cmap) #, node_size = s, node_color = s, alpha = 0.8)
    
    # draw the edges, with color depending on weights
    nx.draw_networkx_edges(V, layout, edge_color = w, edge_cmap = edge_cmap, alpha = 0.8)

    # if desired, add labels
    if lab == True:
        
        # draw labels for the other nodes in black
        nx.draw_networkx_labels(V, layout, labels = labels_others, font_family = "serif", font_size = 40)
    
        # draw labels for the home nodes in red
        nx.draw_networkx_labels(V, layout, labels = labels_homes, font_family = "serif", font_size = 50, font_color = "r")
        
    # show the color palette on the left side of the plot
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm._A = []
    sm = plt.colorbar(sm, aspect = 80)
    
    # add color legend next to the palette
    sm.set_label("Nodes: Genre popularity (average popularity of artists) \n Edges: Genres proximity (share of common artists)", family = "serif", size = 30, rotation = 270)
    sm.ax.tick_params(labelsize = 40, labelcolor = "white")
    
    # if given an artist, the function will print its name on the top left corner, else if given a simple list of genres
    # it will print these genres instead
    if artist is not None:
        plt.title(artist, size = x, loc = "left")
    else:
        plt.title(search_list, size = x, loc = "left")
        
    # remove all axis
    plt.axis("off")
    
    #save the figure
    plt.savefig("sub_network1.png", bbox_inches = "tight", pad_inches = 1)
```

This next simple function allow us to transform a query for an artist to a list of genres for this artist, used in the next and last function.


```python
def artist_to_genres(artist):
    return artists.loc[artists.index == artist, "Genre"].tolist()[0]
```

This last function will first detect if an artist or a list of genres is given. If a list of genres is given, it will run the main function right away. If an artist is given, it will pass the list of genres obtained using the function above before running the main function.


```python
def search_full(genres_search = None, artist_search = None, x = 10, y = 10, lw = 1, k = 1, lab = True):
    
    if genres_search is not None:

        search(genres_search, artist = None, x = x, y = y, lw = lw, k = k, lab = lab)
        
    elif (genres_search is None) & (artist_search is not None):
        
        search(artist_to_genres(artist_search), artist = artist_search, x = x, y = y, lw = lw, k = k, lab = lab)
    
    else:
        print("No terms searched!")
```

Let us put that to use. The band "Jons" belongs to a single genre in our data, so this will produce a simple ego-graph (without edges between non-home nodes). 

How to read the graphs:
* The black node with red label is the genre to which the band belongs;
* The greater the weights (*i.e.* the share of common artists) between any two genres, the more the colour of the edge goes from blue to red, and the closer the shorter the edge (the nodes are closer);
* The greater the size of the genre (*i.e.* its number of artists), the greater the size of the node;
* The greater the genre's popularity (*i.e.* its the average popularity of its artists), the more the colour of the node goes from blue to red.


```python
search_full(artist_search = "Jons" , x = (2*40), y = 40, lw = 40, k = 10, lab = True)
```


<span class="image fit"><img src="{{ "/images/Posts_Images/Part4/part4-3.png" | absolute_url }}" alt="" /></span>


Let us now generate the graph for the band "Triptides", which belongs to more genres in our data. 

The graph becomes more complex, but reads exactly as before. This is were our work gets interesting, because our graph is now composed of several ego-graphs, whose homes (the black nodes) are the genres to which the band Triptides belongs. Nodes present in more than two ego-graphs have degrees greater than or equal to 2.

If you want to explore new genres based on the artist you passed to the function, it would certainly be a good idea to start with genres (nodes) that are located between the black nodes you find the most relevant, or that are close to those you find the most relevant.

For example: I might want to start by genres that have high degrees, because I am most likely to enjoy it, and are quite popular now. Looking at "alternative dance", I was quite impressed to realize I listened to most of its artists in High School! This makes our little system quite relevant, because my musical tastes certainly evolved towards artists that also were influenced by the musical scenes I used to listen. On the other hand, if I want to look for something less popular I might want to have a listen at "popgaze" music, which I found very interesting because I also like the "shoegaze" genre a lot. 


```python
search_full(artist_search = "Triptides" , x = (2*60), y = 60, lw = 20, k = 2, lab = True)
```


<span class="image fit"><img src="{{ "/images/Posts_Images/Part4/part4-4.png" | absolute_url }}" alt="" /></span>


One last example for the road. In this one, we can see how the band Yussef Kamaal is combining two particular musical scenes: one revolving around "jazz funk", "afrobeat" and "contemporary jazz", and the other around "indie jazz".


```python
search_full(artist_search = "Yussef Kamaal" , x = (2*50), y = 50, lw = 20, k = 3, lab = True)
```


<span class="image fit"><img src="{{ "/images/Posts_Images/Part4/part4-5.png" | absolute_url }}" alt="" /></span>

We thus have a really easy to use and fast function that produces pleasant graphs which act as visualizations of the musical identity of a given artist, and could easily be customized and implemented in a production context.
