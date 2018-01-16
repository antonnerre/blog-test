---
layout: post
title:  "Part 2"
date:   2018-01-11
excerpt: "Exploring the New Dataset & Inference "
image: "/images/Posts_Images/Part2/part2.jpg"
---

<span class="image fit"><img src="{{ "/images/Posts_Images/Part2/part2.jpg" | absolute_url }}" alt="" /></span>

Using the dataset built in Part 2, we will try to reveal patterns in the data that could be interesting. This new dataset is already much more interesting, because we have observations at the artist level, each artist belongs to one or more categories (its genres), and we have two numerical values available: the number of followers and the popularity of the artist. We can already expect that the greater the number of follower, the greater the popularity. 

However, there is no direct link between the two, since the popularity value is built by Spotify as the average popularity of a given artist's songs. Each song's popularity is itself a function of two things: the number of times the song was listened, and how recent these listenings were. We will thus check our first intuition that a greater number of follower tends to imply a greater popularity for the artist. We will, afterwards, reveal a new and interesting pattern by considering the genres to which an artist belongs.

## Setting up 

Let us first, as usual, import all modules that are necessary and set our directory.


```ruby
%reset

import pandas as pd
import os
import joypy
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import time
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
import ssl

from scipy import stats
from scipy.stats import linregress
from scipy.stats import pearsonr
from scipy.stats import linregress
from scipy.stats import zscore
from scipy.stats import spearmanr
from statsmodels.compat import lzip

os.chdir("C:/Users/antoi/Documents/Spotify_Project")
```

We also import the dataset that we created at the end of Part 2, as it will be our starting point.


```ruby
full_df = pd.read_csv("Spotify_Full.csv", encoding = "ISO-8859-1")
```

## Cleaning the data

For convenience, we will replace the "+" sign by a whitespace in the Genre column using a list comprehension. We also need to convert the data in the Followers column to floats type for future calculations. The artists with less than 1 follower are removed, because we will use the natural logarithm of the Followers values, whose scale is much larger than Popularity, and we would get ``-Inf`` for artists with no follower. This removes less than 1% of observations which is more than acceptable. 


```python
full_df.Genre = [x.replace("+", " ") for x in full_df.Genre]
full_df.Followers = full_df.Followers.astype(float)

bef = full_df.shape[0]
full_df = full_df[full_df.Followers > 0]
aft = full_df.shape[0]
print(str(100 * (bef - aft) / bef) + "% of observations removed")

full_df["log_Followers"] = np.log(full_df.Followers)
```

    0.6337447114793814% of observations removed
    

For the purpose of cleaner visualization, we also remove a few observations with extremely low number of followers. This last cut removes less than 6% of observations, which is again very reasonable. These observations that were removed for visualization purpose will, however, be used again right after the following two plots.


```python
reg_df = full_df[["Artist", "Popularity", "Followers", "log_Followers"]]

bef = reg_df.shape[0]
reg_df = reg_df[reg_df.Followers >= 10]
aft = reg_df.shape[0]
print(str(100 * (bef - aft) / bef) + "% of observations removed")

reg_df = reg_df.drop_duplicates()
```

    5.934124824800066% of observations removed
    

## Exploring the relationship

Let us start simple and plot a regression line between the Popularity and log_Followers variables, using Seaborn.


```python
%matplotlib inline

sns.set_style("white")
sns.lmplot(x = "log_Followers", y = "Popularity", data = reg_df, size = 9, aspect = 2, line_kws={"alpha" : 0.8, 'color': 'red'},
           scatter_kws={"alpha": 0.05, "color": "blue"})
plt.show()
```

<span class="image fit"><img src="{{ "/images/Posts_Images/Part2/part2-1.png" | absolute_url }}" alt="" /></span>

Looking at the plot above, it seems really reasonable to assume that the variables are linked in the following way: 

$$Popularity = \alpha + \beta \log(Followers) + \epsilon$$ 

$$\Leftrightarrow Followers = e^{-\frac{\alpha}{\beta} + \frac{1}{\beta} Pop + \epsilon}$$ 

Before computing a Pearson correlation coefficient, let us verify whether our data are normally distributed.


```python
%matplotlib inline
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize = (20,5))
sns.distplot(reg_df["Popularity"], ax=ax1)
sns.distplot(reg_df["log_Followers"], ax=ax2)
plt.show()
```


<span class="image fit"><img src="{{ "/images/Posts_Images/Part2/part2-2.png" | absolute_url }}" alt="" /></span>

The data is obviously not normally distributed, so we will use Spearman's rank-order correlation coefficient instead. The coefficient is high and the p-value virtually zero, so the relation we defined earlier is clear.


```python
spearman = spearmanr(reg_df["Popularity"], reg_df["log_Followers"])
print("Spearman rank-order correlation coefficient: " + str(round(spearman[0], 3)))
print("p-value: " + str(spearman[1]))
```

    Spearman rank-order correlation coefficient: 0.91
    p-value: 0.0
    

## Moving further

We showed that, as expected, the popularity and number of followers are indubitably and positively correlated. Now, let us have a first look at how the genres an artist belongs to influence its number of followers. 

We will start by plotting Kernel Density Estimates plots, genre by genre, for the number of followers (which is on the x-axis, so the y-axis is the frequency of such number of followers, binned). To do so, let us compute each genre average popularity using ``groupby``, then add these means to the original dataset. This addition will be used for visualization.

(Notice that from now on we use ``full_df`` and not ``reg_df`` anymore.)


```python
genre_mean = full_df.groupby('Genre', as_index = False)["Popularity"].mean()
full_df = full_df.merge(genre_mean , on=['Genre'])
full_df.rename(columns = {'Popularity_x' : 'Artist Popularity', "Popularity_y" : "Genre Popularity"}, inplace = True)
full_df.head()
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
      <th>Followers</th>
      <th>Artist Popularity</th>
      <th>log_Followers</th>
      <th>Genre Popularity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>pop</td>
      <td>Ed Sheeran</td>
      <td>14003604.0</td>
      <td>99</td>
      <td>16.454825</td>
      <td>69.972769</td>
    </tr>
    <tr>
      <th>1</th>
      <td>pop</td>
      <td>Sam Smith</td>
      <td>4199743.0</td>
      <td>97</td>
      <td>15.250534</td>
      <td>69.972769</td>
    </tr>
    <tr>
      <th>2</th>
      <td>pop</td>
      <td>The Weeknd</td>
      <td>8355421.0</td>
      <td>95</td>
      <td>15.938421</td>
      <td>69.972769</td>
    </tr>
    <tr>
      <th>3</th>
      <td>pop</td>
      <td>Kanye West</td>
      <td>5821775.0</td>
      <td>93</td>
      <td>15.577116</td>
      <td>69.972769</td>
    </tr>
    <tr>
      <th>4</th>
      <td>pop</td>
      <td>Taylor Swift</td>
      <td>7937350.0</td>
      <td>95</td>
      <td>15.887090</td>
      <td>69.972769</td>
    </tr>
  </tbody>
</table>
</div>



## Followers distributions across genres

We are not interested in the values so we will remove the axes and ticks for visibility (and to try to make something aesthetic). We also need some transparency, which is set to 0.3. ``FacetGrid`` allows the usage of ``hue`` with ``map`` and ``distplot``, which is not possible using ``distplot`` on its own. This allows to plot a KDE of the number of followers per artists for each genre, with a color ranging from blue for low genre popularity to red for high genre popularity.


```python
%matplotlib inline

sns.set_style({"axes.facecolor": ".0"})

g = sns.FacetGrid(full_df, hue = "Genre Popularity", size = 30, aspect = 3, palette = "Spectral_r", despine = True)
g = g.map(sns.distplot, "log_Followers", hist = False, kde = True, kde_kws = {"lw": 1.25, "alpha": 0.7})
g = g.set(xticks = [], yticks = [], ylim=(-0.04, None))
g = g.despine(left = True, bottom = True)
g = g.set_xlabels("")
```


<span class="image fit"><img src="{{ "/images/Posts_Images/Part2/part2-3.jpg" | absolute_url }}" alt="" /></span>

There are two clear insights from this:
* The KDEs all appear to be unimodal and have a similar, normal shape;
* A skew is introduced, such that the mode shifts from left to right with an increasing genre popularity (recall that the distributions are for the number of followers at the artist level).

## Feature engineering: diversity

We will create an additional dataset in which one observation corresponds to one artist. In the previous one, we had one observation for each artist and one of its genres. We will count how many genres an artist belongs to and call this its "Diversity". Using "max" for the aggregation of Followers and Artist Popularity is simply to keep a unique value per artist (using "min" would have done the same, since it is repeated for all observations of this artist). Additional modifications are detailled below.


```python
# f is a dict to perform multiple aggregations simultaneously, with groupby
f = {"Followers":["max"], "Artist Popularity":["max"], "Artist":['count']}
grp_df = full_df.groupby('Artist').agg(f)

# renaming the columns, sorting by Popularity, and using the index as a column in the dataset
grp_df.columns = ["Followers", "Artist Popularity", "Diversity"]
grp_df = grp_df.sort_values("Artist Popularity", ascending = False)
grp_df["Artist"] = grp_df.index

# remove odd artists which name contains "?" (errors for artists using symbols in their names)
grp_df = grp_df[~grp_df['Artist'].str.contains("\?")]

# using the natural logarithm of the number of followers will prove usefull
grp_df["log Followers"] = np.log(grp_df["Followers"])

# resetting the index
grp_df.index.name = None
grp_df.index = range(grp_df.shape[0])

grp_df.head()
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
      <th>Followers</th>
      <th>Artist Popularity</th>
      <th>Diversity</th>
      <th>Artist</th>
      <th>log Followers</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1949945.0</td>
      <td>100</td>
      <td>2</td>
      <td>Post Malone</td>
      <td>14.483312</td>
    </tr>
    <tr>
      <th>1</th>
      <td>14004868.0</td>
      <td>99</td>
      <td>4</td>
      <td>Drake</td>
      <td>16.454916</td>
    </tr>
    <tr>
      <th>2</th>
      <td>14003604.0</td>
      <td>99</td>
      <td>1</td>
      <td>Ed Sheeran</td>
      <td>16.454825</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2470277.0</td>
      <td>98</td>
      <td>1</td>
      <td>Ozuna</td>
      <td>14.719841</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2205597.0</td>
      <td>98</td>
      <td>3</td>
      <td>21 Savage</td>
      <td>14.606509</td>
    </tr>
  </tbody>
</table>
</div>




```python
grp_df.to_csv("Artists_Values.csv")
```

## Visualizing diversity

We plot this new feature against the artist's popularity using a violinplot, since the diversity variable is discrete. The violinplot shows the distribution and key statistics for the artists' popularity, given their diversity measure. It is a combination of boxplots and KDE plots. 


```python
%matplotlib inline

sns.set_style({"axes.facecolor": "1"})

with sns.plotting_context("notebook", font_scale=3.5):
    f, axes = plt.subplots(figsize=(54, 18), sharex=True)
    sns.despine(left=True, bottom = True)
    plot2 = sns.violinplot(x = "Diversity", y = "Artist Popularity", data = grp_df[grp_df.Diversity < 35],
                           palette = "Spectral_r")
    axes.set_title("Artists' popularity distributions accross Diversity levels")
```


<span class="image fit"><img src="{{ "/images/Posts_Images/Part2/part2-4.png" | absolute_url }}" alt="" /></span>

Again, two insights appear:
* As the diversity increases, the variance of popularity for artists strongly decreases;
* As the diversity increases, the average popularity increases, then plateaus, and slightly increases again.

## Visualizing the link between Followers and Popularity, genre by genre

The last plot for exploration that we will make represents the regression line between the log of Followers and the Popularity of artists, genre by genre. The greater the genre popularity the more the line goes from blue to red.


```python
%matplotlib inline

full_df = full_df.sort_values("Genre Popularity", ascending = True)

sns.set_style("white", {"axes.facecolor": ".0"})

g = sns.lmplot(x = "log_Followers", y = "Artist Popularity", hue = "Genre Popularity",
               palette = "Spectral_r", data = full_df, scatter_kws = {"s": 0},
               line_kws = {"alpha": 0.3}, legend = False, size = 50, aspect = 2, ci = None)

g = g.set(xticks = [], yticks = [])
g = g.despine(left = True, bottom = True)
g = g.set_xlabels("")
g = g.set_ylabels("")
```


<span class="image fit"><img src="{{ "/images/Posts_Images/Part2/part2-5.jpg" | absolute_url }}" alt="" /></span>

This probably reveals the most striking and unexpected pattern:
* As the genre popularity increases, the slope of the regression line increases as well.

How to interpret this ? A greater slope means more popularity with less followers. There are two possible explanations:
* A greater popularity comes with more listenings. Assuming the average listener of a given genre listens to as much music as the average listener of any other genre, it could mean that the listeners of the most popular genres listen to these genres exclusively, while listeners of less popular genres listen to more different genres.
* An other possible explanation could be that users listening to the most popular genres are less "attached" to these genres' artists because they do not follow them as much: for instance, they could listen to popular playlists without having a particular interest for the artists in these playlist, while listeners of less popular genres could have more interest in the artists.

Let us dive a bit deeper in regarding this last point by doing some inference.

# Statistical Inference

After having shown the existence of the previous phenomenon, let us try to assess how much it depends on popularity and, possibly, additional variables.

## Gathering the data

The next table gathers the data we created earlier and will be aggregated in various ways just after. For now we simply merge the tables.


```python
artist = full_df.merge(grp_df[['Diversity', "Artist"]] , on=['Artist'])
artist = artist.sort_values("Genre Popularity", ascending = False)
artist.head()
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
      <th>Followers</th>
      <th>Artist Popularity</th>
      <th>log_Followers</th>
      <th>Genre Popularity</th>
      <th>Diversity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>335821</th>
      <td>pop</td>
      <td>Ed Sheeran</td>
      <td>14003604.0</td>
      <td>99</td>
      <td>16.454825</td>
      <td>69.972769</td>
      <td>1</td>
    </tr>
    <tr>
      <th>318746</th>
      <td>pop</td>
      <td>Greyson Chance</td>
      <td>184399.0</td>
      <td>61</td>
      <td>12.124857</td>
      <td>69.972769</td>
      <td>5</td>
    </tr>
    <tr>
      <th>318775</th>
      <td>pop</td>
      <td>Shane Harper</td>
      <td>92748.0</td>
      <td>52</td>
      <td>11.437641</td>
      <td>69.972769</td>
      <td>4</td>
    </tr>
    <tr>
      <th>318768</th>
      <td>pop</td>
      <td>Lucy Hale</td>
      <td>170443.0</td>
      <td>53</td>
      <td>12.046156</td>
      <td>69.972769</td>
      <td>4</td>
    </tr>
    <tr>
      <th>303201</th>
      <td>pop</td>
      <td>Kurt Hugo Schneider</td>
      <td>149922.0</td>
      <td>70</td>
      <td>11.917870</td>
      <td>69.972769</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>



Now comes the aggregation part: we will go back to the genre level, but this time we will have a lot more of information.

The aggregations are detailed below. As a summary, we create a brand new table, ``genre``, whose columns all result from the aggregation of the previous table. In particular, we have:
* Concentration: this variable is the slope of each regression shown in the previous plot. We will call it "concentration" because a greater slope means that users' listenings are more concentrated towards this genre (1st explanation above). To do so, we can ``groupby`` genre and compute the slope using a ``lambda`` function;
* Popularity: we group by genre and take the average popularity of artists in the genre;
* Followers: we group by genre and sum the number of followers of artists in the genre;
* Openness: we group by genre and take the average diversity of artists in the genre. If artists of a genre have a large diversity, this genre is "open" to other genres.
* Size: we group by genre and count the number of artists in the genre.


```python
genre = pd.DataFrame()

# Regression slopes
genre["Concentration"] = full_df.groupby('Genre').apply(lambda v: linregress(v["log_Followers"] , v["Artist Popularity"])[0])

# Average popularity of artists representing the genre
genre["Popularity"] = artist.groupby('Genre')["Artist Popularity"].mean()

# Sum of users following artists representing the genre
genre["Followers"] = artist.groupby('Genre')["Followers"].sum()

# Average number of genres the artists representing the genre belong to
# -> If most artist in genre x alost belong in a lot of different genres, genre x will have a high openness
genre["Openness"] = artist.groupby("Genre")["Diversity"].mean()

genre["Size"] = full_df.groupby("Genre").count()["Artist"]

# reset the index
genre["Genre"] = genre.index
genre.index.name = None
genre.index = range(genre.shape[0])
genre.head()
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
      <th>Concentration</th>
      <th>Popularity</th>
      <th>Followers</th>
      <th>Openness</th>
      <th>Size</th>
      <th>Genre</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7.462457</td>
      <td>23.712121</td>
      <td>2579807.0</td>
      <td>1.646465</td>
      <td>198</td>
      <td>a cappella</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3.999776</td>
      <td>5.510345</td>
      <td>270059.0</td>
      <td>2.372414</td>
      <td>145</td>
      <td>abstract</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6.494923</td>
      <td>18.622857</td>
      <td>410221.0</td>
      <td>1.862857</td>
      <td>175</td>
      <td>abstract beats</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7.073912</td>
      <td>25.755906</td>
      <td>2323195.0</td>
      <td>2.299213</td>
      <td>257</td>
      <td>abstract hip hop</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.201111</td>
      <td>6.300000</td>
      <td>75865.0</td>
      <td>1.406250</td>
      <td>160</td>
      <td>abstract idm</td>
    </tr>
  </tbody>
</table>
</div>




```python
artist = artist.merge(genre[['Size', "Genre"]] , on=['Genre'])
artist.head()
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
      <th>Followers</th>
      <th>Artist Popularity</th>
      <th>log_Followers</th>
      <th>Genre Popularity</th>
      <th>Diversity</th>
      <th>Size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>pop</td>
      <td>Ed Sheeran</td>
      <td>14003604.0</td>
      <td>99</td>
      <td>16.454825</td>
      <td>69.972769</td>
      <td>1</td>
      <td>661</td>
    </tr>
    <tr>
      <th>1</th>
      <td>pop</td>
      <td>Greyson Chance</td>
      <td>184399.0</td>
      <td>61</td>
      <td>12.124857</td>
      <td>69.972769</td>
      <td>5</td>
      <td>661</td>
    </tr>
    <tr>
      <th>2</th>
      <td>pop</td>
      <td>Shane Harper</td>
      <td>92748.0</td>
      <td>52</td>
      <td>11.437641</td>
      <td>69.972769</td>
      <td>4</td>
      <td>661</td>
    </tr>
    <tr>
      <th>3</th>
      <td>pop</td>
      <td>Lucy Hale</td>
      <td>170443.0</td>
      <td>53</td>
      <td>12.046156</td>
      <td>69.972769</td>
      <td>4</td>
      <td>661</td>
    </tr>
    <tr>
      <th>4</th>
      <td>pop</td>
      <td>Kurt Hugo Schneider</td>
      <td>149922.0</td>
      <td>70</td>
      <td>11.917870</td>
      <td>69.972769</td>
      <td>6</td>
      <td>661</td>
    </tr>
  </tbody>
</table>
</div>




```python
genre.to_csv("Spotify_Genre.csv")
artist.to_csv("Spotify_Artist.csv")
```

We can start by looking at the correlation between $$Popularity$$ and $$\log(Followers)$$. We look at their distribution to choose the correlation coefficient to use.


```python
%matplotlib inline
sns.set_style("white", {"axes.facecolor": "1"})
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize = (20,5))
sns.distplot(genre["Popularity"], ax=ax1)
sns.distplot(np.log(genre["Followers"]), ax=ax2)
plt.show()
```


<span class="image fit"><img src="{{ "/images/Posts_Images/Part2/part2-6.png" | absolute_url }}" alt="" /></span>

Again, since they are not normally distributed, we will use Spearman's coefficient, which shows a significant and strong positive correlation.


```python
spearman = spearmanr(genre["Popularity"], np.log(genre["Followers"]))
print("Spearman rank-order correlation coefficient: " + str(round(spearman[0], 3)))
print("p-value: " + str(spearman[1]))
```

    Spearman rank-order correlation coefficient: 0.931
    p-value: 0.0
    

## Building the model

To explain the Concentration, we of course use the variable Popularity, but it is important to account for other variables that could lead to ommited variables bias if not included. In particular, it is very possible that:
* the greater the size of a genre the greater its concentration, since there are more artists to listen to;
* the greater the openness of a genre the greater its concentration, since its artists already cover a larger range of diversity and you might not feel the need to listen to something significantly different.

We thus include these variables in the first Ordinary Least Squares regression, including quadratic and cubic terms for $$\log(Popularity)$$ to capture non-linearity.

Followers will not be included, since it will create obvious multicolinearity issues with Popularity. A 2-Stages Least Squares model could have been built, but Followers is itself strongly correlated with Concentration, making it a poor instrument. Moreover, we demean POPULARITY to avoid multicolinearity with its quadratic and cubic term.


```python
genre_reg = pd.DataFrame()
genre_reg["POPULARITY"] = np.log(genre["Popularity"]) - np.mean(np.log(genre["Popularity"]))
genre_reg["POPULARITY_2"] = genre_reg["POPULARITY"]**2
genre_reg["POPULARITY_3"] = genre_reg["POPULARITY"]**3
genre_reg["CONCENTRATION"] = genre["Concentration"]
genre_reg["OPENNESS"] = genre["Openness"]
genre_reg["SIZE"] = genre["Size"]

results = smf.ols('CONCENTRATION ~ POPULARITY + POPULARITY_2 + POPULARITY_3 + OPENNESS + SIZE', data = genre_reg).fit()
print(results.summary())
alpha = 0.01
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:          CONCENTRATION   R-squared:                       0.841
    Model:                            OLS   Adj. R-squared:                  0.841
    Method:                 Least Squares   F-statistic:                     1608.
    Date:                Thu, 04 Jan 2018   Prob (F-statistic):               0.00
    Time:                        16:40:38   Log-Likelihood:                -1543.3
    No. Observations:                1520   AIC:                             3099.
    Df Residuals:                    1514   BIC:                             3131.
    Df Model:                           5                                         
    Covariance Type:            nonrobust                                         
    ================================================================================
                       coef    std err          t      P>|t|      [0.025      0.975]
    --------------------------------------------------------------------------------
    Intercept        5.7910      0.058    100.653      0.000       5.678       5.904
    POPULARITY       1.7601      0.044     40.142      0.000       1.674       1.846
    POPULARITY_2    -0.6337      0.044    -14.304      0.000      -0.721      -0.547
    POPULARITY_3    -0.2552      0.019    -13.390      0.000      -0.293      -0.218
    OPENNESS         0.1050      0.009     11.880      0.000       0.088       0.122
    SIZE            -0.0002      0.000     -0.844      0.399      -0.001       0.000
    ==============================================================================
    Omnibus:                      159.768   Durbin-Watson:                   1.842
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              511.073
    Skew:                          -0.515   Prob(JB):                    1.05e-111
    Kurtosis:                       5.647   Cond. No.                         968.
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    

## Model adjustment
It turns out that the Size variable is not significant so we will remove it from the model.


```python
results = smf.ols('CONCENTRATION ~ POPULARITY + POPULARITY_2 + POPULARITY_3 + OPENNESS', data = genre_reg).fit()
print(results.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:          CONCENTRATION   R-squared:                       0.841
    Model:                            OLS   Adj. R-squared:                  0.841
    Method:                 Least Squares   F-statistic:                     2010.
    Date:                Thu, 04 Jan 2018   Prob (F-statistic):               0.00
    Time:                        16:40:38   Log-Likelihood:                -1543.7
    No. Observations:                1520   AIC:                             3097.
    Df Residuals:                    1515   BIC:                             3124.
    Df Model:                           4                                         
    Covariance Type:            nonrobust                                         
    ================================================================================
                       coef    std err          t      P>|t|      [0.025      0.975]
    --------------------------------------------------------------------------------
    Intercept        5.7506      0.032    180.621      0.000       5.688       5.813
    POPULARITY       1.7401      0.037     47.172      0.000       1.668       1.812
    POPULARITY_2    -0.6521      0.039    -16.936      0.000      -0.728      -0.577
    POPULARITY_3    -0.2598      0.018    -14.247      0.000      -0.296      -0.224
    OPENNESS         0.1038      0.009     11.903      0.000       0.087       0.121
    ==============================================================================
    Omnibus:                      158.281   Durbin-Watson:                   1.843
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):              507.861
    Skew:                          -0.509   Prob(JB):                    5.24e-111
    Kurtosis:                       5.642   Cond. No.                         11.1
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    

All of our variables are very significant, and our intuition on Openness turned out true since its coefficient is positive. For Popularity, it will be a little bit more complicated and we will tackle that right after checking that all assumptions for OLS are verified.

## Checking OLS assumptions
### Outliers and their influence
Using Bonferroni's method for outliers detection, we remove genres with p-values below 1, and run the regression again. 8 outliers were removed. All statistics improve and variables do not change sign, so our model appears robust. This conclusion can also be drawn from when we remove the Size variable, which did not significantly change other variables' p-value or coefficient.


```python
outliers = results.outlier_test()
outliers = outliers.sort_values("bonf(p)")
genre_reg = genre_reg.drop(outliers[outliers["bonf(p)"] < 1].index)

results = smf.ols('CONCENTRATION ~ POPULARITY + POPULARITY_2 + POPULARITY_3 + OPENNESS', data = genre_reg).fit()
print(results.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:          CONCENTRATION   R-squared:                       0.857
    Model:                            OLS   Adj. R-squared:                  0.857
    Method:                 Least Squares   F-statistic:                     2260.
    Date:                Thu, 04 Jan 2018   Prob (F-statistic):               0.00
    Time:                        16:40:39   Log-Likelihood:                -1453.9
    No. Observations:                1512   AIC:                             2918.
    Df Residuals:                    1507   BIC:                             2944.
    Df Model:                           4                                         
    Covariance Type:            nonrobust                                         
    ================================================================================
                       coef    std err          t      P>|t|      [0.025      0.975]
    --------------------------------------------------------------------------------
    Intercept        5.7738      0.030    190.828      0.000       5.714       5.833
    POPULARITY       1.7803      0.035     50.720      0.000       1.711       1.849
    POPULARITY_2    -0.6027      0.037    -16.351      0.000      -0.675      -0.530
    POPULARITY_3    -0.2452      0.017    -14.114      0.000      -0.279      -0.211
    OPENNESS         0.0929      0.008     11.176      0.000       0.077       0.109
    ==============================================================================
    Omnibus:                       18.506   Durbin-Watson:                   1.808
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):               28.940
    Skew:                          -0.079   Prob(JB):                     5.20e-07
    Kurtosis:                       3.659   Cond. No.                         11.2
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    

## Checking residuals normality
We perform a simple Lilliefors test for normality. We use this test because we do not have a priori idea about the parameters of the normal distribution to test our residuals against. The test does not allow us to reject the hypothesis of normally distributed residuals.


```python
%matplotlib inline
sns.distplot(results.resid, kde = True)
if sms.lilliefors(results.resid)[1] > alpha:
    print("Lilliefors test: cannot reject the null hypothesis of normality." )
else:
    print("Lilliefors test: reject the null hypothesis of normality.")
```

    Lilliefors test: cannot reject the null hypothesis of normality.
    

![normality]({{ "./images/Posts_Images/Part2/part2-7.png" | absolute_url }})

## Checking multicolinearity
We manually compute the Variance Inflation Factors of our independent variables since non function in any python module seems to exist to do this. The VIFs are computed as $$\frac{1}{1 - R_j^2}$$, where $$R_j^2$$ is the coefficient of determination of the linear regression of the independent variable $$j$$ against all other independent variables. The resulting VIFs show no alarming multicolinearity as they all lie below 7.


```python
results_pop = smf.ols('POPULARITY ~ POPULARITY_2 + POPULARITY_3 + OPENNESS', data = genre_reg).fit()
results_pop_2 = smf.ols('POPULARITY_2 ~  POPULARITY + POPULARITY_3 + OPENNESS', data = genre_reg).fit()
results_pop_3 = smf.ols('POPULARITY_3 ~ POPULARITY + POPULARITY_2 + OPENNESS', data = genre_reg).fit()
results_open = smf.ols('OPENNESS ~ POPULARITY + POPULARITY_2 + POPULARITY_3', data = genre_reg).fit()

vif_pop = 1 / (1 - results_pop.rsquared)
vif_pop_2 = 1 / (1 - results_pop_2.rsquared)
vif_pop_3 = 1 / (1 - results_pop_3.rsquared)
vif_open = 1 / (1 - results_open.rsquared)

print("POPULARITY VIF: " + str(round(vif_pop, 2)) + "\n" +
      "POPULARITY_2 VIF: " + str(round(vif_pop_2, 2)) + "\n" +
      "POPULARITY_3 VIF: " + str(round(vif_pop_3, 2)) + "\n" +
      "OPENNESS VIF: " + str(round(vif_open, 2)))
```

    POPULARITY VIF: 3.18
    POPULARITY_2 VIF: 5.25
    POPULARITY_3 VIF: 6.87
    OPENNESS VIF: 1.85
    

## Checking homoscedasticity
This is where we face a small issue. We use Breusch-Pagan and White tests to check for homoscedasticity. If at least one rejects the null hypothesis of homoscedasticity, we will consider that we have heteroscedasticity.


```python
%matplotlib inline

white_test = sms.het_white(results.resid, results.model.exog)[1] < alpha
BP_test = sms.het_breuschpagan(results.resid, results.model.exog)[1] < alpha

if white_test or BP_test:
    print("White test and Breusch-Pagan test: reject the null hypothesis of homoscedasticity.")
else:
    print("White test and Breusch-Pagan test: cannot reject the null hypothesis of homoscedasticity.")
```

    White test and Breusch-Pagan test: reject the null hypothesis of homoscedasticity.
    

At least one of the tests rejects the null hypothesis. Let us have a look at the residuals. There appears to be a "cone", from left to right, but nothing particularly striking.


```python
diagn_df = pd.DataFrame()
diagn_df["Residuals"], diagn_df["Fitted"] = results.resid, results.fittedvalues

sns.lmplot(x = "Fitted", y = "Residuals", data = diagn_df, order = 1, size = 5, aspect = 2,
           line_kws={'color': sns.xkcd_rgb["pale red"]}, scatter_kws={"alpha": 0.3, "color": sns.xkcd_rgb["water blue"]})
plt.show()
```


<span class="image fit"><img src="{{ "/images/Posts_Images/Part2/part2-8.png" | absolute_url }}" alt="" /></span>

Since our model will only be used for inference and not for prediction, we are fine, as OLS estimators remain unbiased and consistent under heteroskedasticity. There is thus no need to use Weighted Least Squares or Robust Least Squares.

They are, however, **unefficient**.

For this reason, we need to use robust standard errors in our regression, to account for the potential heteroscedasticity. This is done within the ``.fit()`` function, and we can use the HC0. 


```python
results = smf.ols('CONCENTRATION ~ POPULARITY + POPULARITY_2 + POPULARITY_3 + OPENNESS', data = genre_reg).fit(cov_type = "HC0")
print(results.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:          CONCENTRATION   R-squared:                       0.857
    Model:                            OLS   Adj. R-squared:                  0.857
    Method:                 Least Squares   F-statistic:                     3016.
    Date:                Thu, 04 Jan 2018   Prob (F-statistic):               0.00
    Time:                        16:40:40   Log-Likelihood:                -1453.9
    No. Observations:                1512   AIC:                             2918.
    Df Residuals:                    1507   BIC:                             2944.
    Df Model:                           4                                         
    Covariance Type:                  HC0                                         
    ================================================================================
                       coef    std err          z      P>|z|      [0.025      0.975]
    --------------------------------------------------------------------------------
    Intercept        5.7738      0.033    176.898      0.000       5.710       5.838
    POPULARITY       1.7803      0.043     41.756      0.000       1.697       1.864
    POPULARITY_2    -0.6027      0.043    -13.988      0.000      -0.687      -0.518
    POPULARITY_3    -0.2452      0.026     -9.423      0.000      -0.296      -0.194
    OPENNESS         0.0929      0.010      9.386      0.000       0.073       0.112
    ==============================================================================
    Omnibus:                       18.506   Durbin-Watson:                   1.808
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):               28.940
    Skew:                          -0.079   Prob(JB):                     5.20e-07
    Kurtosis:                       3.659   Cond. No.                         11.2
    ==============================================================================
    
    Warnings:
    [1] Standard Errors are heteroscedasticity robust (HC0)
    

Naturaly, our standard errors increased accordingly. The confidence intervals for the coefficients changed slightly. 

## Summarizing the model
In the end, we can be quite confident in the model we estimated for inference. 

We estimated the following:

$$ CONC(POP, OPEN) = \alpha + \beta_1 \times POP + \beta_2 \times POP^2 + \beta_3 \times POP^3 + \beta_4 \times OPEN + \epsilon$$

where

$$POP = popularity - \overline{popularity}$$

with

$$\overline{popularity} = \frac{1}{n} \sum_{i = 1}^{n} popularity_i$$

and

$$popularity = \log(Popularity)$$

## Is Concentration always increasing with Popularity ?
Finaly, let us verify our assumption that a genre concentration increases with its popularity. We will look at the gradient of $$CONC(POP, OPEN)$$ and check whether it is positive.

$$CONC'(POP) = \frac{\partial CONC(POP, OPEN)}{\partial POP} = \beta_1 + \beta_2 \times POP + \beta_3 \times POP^2$$

We can plot $$CONC'(POP)$$.


```python
%matplotlib inline

### Preparing data for the plot

# Regression coefficients
coeffs = results.params

# Gradient
x_full = np.arange(min(genre_reg["POPULARITY"]) - 1, max(genre_reg["POPULARITY"]) + 1, 0.001)
x = np.arange(min(genre_reg["POPULARITY"]), max(genre_reg["POPULARITY"]), 0.001)
y_full = coeffs[1] + coeffs[2] * x_full + coeffs[3] * x_full**2
y = coeffs[1] + coeffs[2] * x + coeffs[3] * x**2

# x = 0 axis
y2 = 0 * x_full

# min and max axes
y3 = np.arange(-1, 2.5, 0.001)
x3 = [min(genre_reg["POPULARITY"])] * len(y3)
x4 = [max(genre_reg["POPULARITY"])] * len(y3)

### Plotting
sns.set(style="white")
plt.rcParams['font.family'] = 'serif'
fig, ax = plt.subplots(figsize = (15, 8))
ax.plot(x_full, y_full, color=sns.xkcd_rgb["water blue"], linewidth = 1)
ax.plot(x, y, color=sns.xkcd_rgb["red"], linewidth = 1)
ax.plot(x_full, y2, color = sns.xkcd_rgb["almost black"], dashes=[10, 5, 10, 5], linewidth = 1)
ax.plot(x3, y3, color = sns.xkcd_rgb["grey"], dashes=[5, 5, 5, 5], linewidth = 0.75)
ax.plot(x4, y3, color = sns.xkcd_rgb["grey"], dashes=[5, 5, 5, 5], linewidth = 0.75)
ax.axvspan(min(genre_reg["POPULARITY"]), max(genre_reg["POPULARITY"]), alpha=0.1, color='gray', label = "POP's range")
ax.legend(loc = "center right")
ax.set_xlabel("POP")
ax.set_ylabel("CONC'(POP)")
sns.despine()
```


<span class="image fit"><img src="{{ "/images/Posts_Images/Part2/part2-9.png" | absolute_url }}" alt="" /></span>

We can see that on the range of values $$POP$$ can take (i.e. from $$Popularity$$ that belongs to $$[0 , 100]$$), the gradient is always positive, though increasing then decreasing. 

   **We can thus conclude that a genre's concentration increases with its popularity.**

### Saving the datasets for further use


```python
genre.to_csv("Spotify_Genre_bis.csv")
artist.to_csv("Spotify_Artist_bis.csv")
```

# Summary

In this Part, we did the following:
* We checked our intuition that Popularity and Followers should be positively correlated, using visualization and the appropriate statistical tools;
* We revealed three additional and interesting patterns, through feature engineering;
* We carefully analyzed the relationship between what we called Concentration and Popularity, using statistical inference.
* We verified the assumptions of our model, and adapted it accordingly.

# BONUS: the Unknown Pleasures of Kernel Density Estimates

What is better than a joyplot to visualize the musical landscape ? This is basically the same plot as the first one: each line represents the Kernel Density Estimate of the distribution of followers (log) per artists of a given genre. Notice the shift of the distributions' modes from left to right as you scroll down: instead of being colored from blue to red, genres were ordered from top to bottom by their popularity.

(In case you don't know Unknown Pleasures by Joy Division yet, have a look at its album artwork and, most importantly, have a first listen: https://www.allmusic.com/album/unknown-pleasures-mw0000202764 . This last plot is largely inspired by the work of user ``sbebo`` on Github and its package ``joypy``, and adapted to look a bit more like the album cover: the KDEs are not transparent, and the figure is more centered. Also, we are plotting music!)


```python
%matplotlib inline

fig, axes = joypy.joyplot(artist[artist["Size"] > 50] , by = "Genre Popularity", column = "log_Followers", ylabels = False,
                          xlabels = False, grid = False, background = 'black', linecolor = "white", linewidth = 2, ylim = "own",
                          colormap = mpl.colors.ListedColormap(['black']), legend = False, overlap = 1.5, figsize = (16,112),
                          kind = "kde", bins = 25, x_range = [-9, 23])

plt.subplots_adjust(left = 0, right = 1, top = 1, bottom=0)
for a in axes[:-1]:
    a.set_xlim([-25,39])
```


<span class="image fit"><img src="{{ "/images/Posts_Images/Part2/part2-10.png" | absolute_url }}" alt="" /></span>