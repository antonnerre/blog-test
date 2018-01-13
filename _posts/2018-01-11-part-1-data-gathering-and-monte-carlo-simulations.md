---
layout: post
title:  "Part 1"
date:   2018-01-11
excerpt: "Data gathering & Monte Carlo Simulations "
image: "./images/Posts_Images/Part1/part1.png"
---

## Setting up

As expected, we will first need to call the Spotify API using ``spotipy`` to build our base dataset. Have a look at the modules we will use in this first part of the serie. If you wish to try the code on your machine, don't forget to set you own directory.


{% highlight ruby %}
# Deletes all variables
%reset

# Imports necessary modules
import spotipy
import spotipy.util as util
import spotipy.oauth2 as oauth2
import random
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy
import scipy.stats as stats
import math
from urllib3.exceptions import HTTPError
import sys as sys
import ssl

# Sets the directory
os.chdir("C:/Users/antoi/Documents/Spotify_Project")
{% endhighlight %}

    Once deleted, variables cannot be recovered. Proceed (y/[n])? y
    

Now let us set up the web API authorization to access Spotify data. You can easily generate your own client's ID and Secret at the Spotify for Developers website: https://beta.developer.spotify.com

We first obtain the credentials, which are used to get the access token that ``spotipy`` will use. 


```python
creds = oauth2.SpotifyClientCredentials("43ff17941daa47899220a15f54d2301b", "ddadfa139c484861bffd5eda76f8f521")
token = creds.get_access_token()
spotify = spotipy.Spotify(token)
```

We will use a text file that lists all music genres one can find on Spotify. This text file simply consists of the list from Every Noise at Once (http://everynoise.com/everynoise1d.cgi) that you can copy and paste in a text editor. We create ``Genres_0`` from this text file, where a new line means a new entry. ``Genres`` will be the same but with a "+" sign instead of a whitespace, which will be useful when we make our queries to the API. We can quickly do this using a list comprehension.


```python
with open("Genres.txt") as f:
    Genres_0 = f.read().splitlines()
    
Genres = [x.replace(" ", "+") ffor x in Genres_0]
```

We also need to create an "empty" dataframe, which for now consists of the list of Genres in one column and zeros in the other. We will feed the second column with the number of artists representing the given genre.


```python
import pandas as pd
df = pd.DataFrame({'Genre': Genres_0, "Number of artists":0})
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
      <th>Number of artists</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>pop</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>dance pop</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>pop rap</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>rap</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>post-teen pop</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## Genre level queries and analysis
We can now make our first queries. Below is an example of what gets returned when we call a search for the genre "Indie Jazz".


```python
spotify.search(q = "genre:" + "indie+jazz", type = 'artist', limit = 1, offset = 0)
```




    {'artists': {'href': 'https://api.spotify.com/v1/search?query=genre%3Aindie%2Bjazz&type=artist&offset=0&limit=1',
      'items': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/4gHcu2JoaXJ0mV4aNPCd7N'},
        'followers': {'href': None, 'total': 13415},
        'genres': ['indie jazz'],
        'href': 'https://api.spotify.com/v1/artists/4gHcu2JoaXJ0mV4aNPCd7N',
        'id': '4gHcu2JoaXJ0mV4aNPCd7N',
        'images': [{'height': 640,
          'url': 'https://i.scdn.co/image/61b623dc38e24dae5135235522e639f5c9a1dbd9',
          'width': 640},
         {'height': 320,
          'url': 'https://i.scdn.co/image/27c7e695fc632bd76e11d4c8a32886f635e3ac13',
          'width': 320},
         {'height': 160,
          'url': 'https://i.scdn.co/image/3cc7417ab387e1525cdae2047f53a04cfe3fda27',
          'width': 160}],
        'name': 'Benny Sings',
        'popularity': 61,
        'type': 'artist',
        'uri': 'spotify:artist:4gHcu2JoaXJ0mV4aNPCd7N'}],
      'limit': 1,
      'next': 'https://api.spotify.com/v1/search?query=genre%3Aindie%2Bjazz&type=artist&offset=1&limit=1',
      'offset': 0,
      'previous': None,
      'total': 320}}



We can see data about an artist belonging to this genre (here, "BadBadNotGood). What we would like to do for now is to replace the zeros in ``df`` by the last number one can find in the previous query: the total number of "pages" we can look at in this genre, that is, the total number of artists in this genre. It can be accessed selecting "artists" then "total" on the result of the query. We will thus use a ``for`` loop, where for each genre in the list ``Genres`` we will make a query then replace the corresponding zero by the total number of artists in this genre.


```python
for genre in Genres:
    results = spotify.search(q = "genre:" + genre, type = 'artist', limit = 1, offset = 1)
    df.loc[Genres.index(genre), "Number of artists"] = results["artists"]["total"]
```    

Let us see what the result looks like.


```python
df.head(10)
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
      <th>Number of artists</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>pop</td>
      <td>661</td>
    </tr>
    <tr>
      <th>1</th>
      <td>dance pop</td>
      <td>756</td>
    </tr>
    <tr>
      <th>2</th>
      <td>pop rap</td>
      <td>632</td>
    </tr>
    <tr>
      <th>3</th>
      <td>rap</td>
      <td>480</td>
    </tr>
    <tr>
      <th>4</th>
      <td>post-teen pop</td>
      <td>433</td>
    </tr>
    <tr>
      <th>5</th>
      <td>tropical house</td>
      <td>556</td>
    </tr>
    <tr>
      <th>6</th>
      <td>rock</td>
      <td>623</td>
    </tr>
    <tr>
      <th>7</th>
      <td>modern rock</td>
      <td>672</td>
    </tr>
    <tr>
      <th>8</th>
      <td>trap music</td>
      <td>456</td>
    </tr>
    <tr>
      <th>9</th>
      <td>dwn trap</td>
      <td>412</td>
    </tr>
  </tbody>
</table>
</div>



The distribution of the number of artists across genres has the following distribution, which seems really close to a Gamma distribution. 


```python
plt.figure()
sns.distplot(np.array(df["Number of artists"]), bins = 50, label = "Empirical Distribution")
plt.xlabel("Number of Artists")
plt.legend()
plt.show()
```


![1st distribution]({{ "./images/Posts_Images/Part1/part1-1.png" | absolute_url }})

Let us thus fit the data to a Gamma distribution.


```python
random.seed(33)
fit_alpha, fit_loc, fit_beta = stats.gamma.fit(df["Number of artists"])
gam = stats.gamma.rvs(fit_alpha, fit_loc, scale = fit_beta, size = len(df["Number of artists"]))
```

We can visually compare the two: they indeed appear very close.


```python
plt.figure()
sns.distplot(np.array(gam), bins = 50, label = "Gamma Distribution")
sns.distplot(np.array(df["Number of artists"]), bins = 50, label = "Empirical Distribution")
plt.xlabel("Number of Artists")
plt.legend()
plt.show()
```


![2nd distribution]({{ "./images/Posts_Images/Part1/part1-2.png" | absolute_url }})

Let us try to assess the goodness-of-fit of the Gamma distribution to the data, by running a Kolmogorov-Smirnov test. The parameters for the Gamma distribution are obtained by fitting the Gamma distribution to the data, using Maximum Likelihood Estimation. We randomly sample from the column with the number of artists (1000 observations), because there is no point in using the full data for these kind of tests that get way too sensitive when the sample size is too large.


```python
sample = np.random.choice(df["Number of artists"], 1000)

distribution = getattr(stats, "gamma")
parameters = distribution.fit(sample)
stats.kstest(sample, "gamma", args = parameters)
```




    KstestResult(statistic=0.035681426325871679, pvalue=0.15302864239908143)



### Monte Carlo for the Kolmogorov-Smirnov test

There is, however, one issue: the Kolmogorov-Smirnov test cannot be used when the parameters of the tested distribution were estimated. Instead, they must be completely specified beforehand.

What was done just before is thus not enough, and we actually need to perform some (basic) Monte Carlo simulations (I would avoid speaking about parametric bootstrapping here, because even if we use the data to fit the parameters, we do not actually sample from it at each iteration of the simulation).

This is quite close to a Lilliefors test, but using a Gamma distribution instead of a Normal or Exponential.

The p-value will correspond to how often the KS statistic was at least as high as the one obtained in the previous test, when we perform the KS-test on random Gamma data (generated with the estimated parameters) against a Gamma distribution (with the same parameters).


```python
mc = []

# Notice that we will choose the 0th element from the kstest object to obtain the statistic and check
# how often it is BELOW those from the MC
for i in range(1000):
    fit_alpha, fit_loc, fit_beta = stats.gamma.fit(sample)
    gam = stats.gamma.rvs(fit_alpha, fit_loc, fit_beta, size = len(sample))
    mc.append(stats.kstest(gam, "gamma", args = (fit_alpha, fit_loc, fit_beta))[0])
    
p = np.mean(mc >= stats.kstest(sample, "gamma", args = (fit_alpha, fit_loc, fit_beta))[0])
np.round(p, decimals = 4)
```




    0.13300000000000001



Conversely, it could be how often the p-value was at least as low as the one obtained in the previous test. We should observe a close p-value by doing as follows.


```python
mc = []

# Notice that here, compared to what we did above, we will choose the 1st element (not 0th) to obtain
# the p-value (not the statistic) and check how often it is ABOVE those from the MC
for i in range(1000):
    fit_alpha, fit_loc, fit_beta = stats.gamma.fit(sample)
    gam = stats.gamma.rvs(fit_alpha, fit_loc, fit_beta, size = len(sample))
    mc.append(stats.kstest(gam, "gamma", args = (fit_alpha, fit_loc, fit_beta))[1])
    
p = np.mean(mc <= stats.kstest(sample, "gamma", args = (fit_alpha, fit_loc, fit_beta))[1])
np.round(p, decimals = 4)
```




    0.14699999999999999



This does not change the p-value much from the first, incorrect test; but we can now be assured that we cannot reject the hypothesis that our data comes from a Gamma distribution.

## Artist level queries

In the previous section that was more of an introduction to the API, the dataset was limited and there was not much to learn beyond the fact that the number of artists per genres is most likely Gamma distributed. We will create a new dataset, where one observation corresponds to one artist rather than a genre. There must be some additional and interesting things to discover.

Let us start by defining an empty dataframe, with columns ``Genre``, ``Artist``, ``Followers`` and ``Popularity``. 


```python
df1 = pd.DataFrame(columns = ["Genre", "Artist", "Followers", "Popularity"])
```

We must define two functions that will prove extremely useful. The first one is the main one: ``add_row()``. This function will add the appropriate values at row ``i``. Artist number ``n`` of the current genre defines the offset for the query which, because we call one artist per query (``limit = 1``), also corresponds to the page we are calling. 


```python
def add_row():
    
    # We will need this i to keep on increasing, even when we move to a new genre
    global i
    
    # This is the same query we saw earlier, except now we will go beyond artist (or page) 0
    results = spotify.search(q = 'genre:' + genre, type = 'artist', limit = 1, offset = n)
    
    # Feed the dataframe with values obtained with the query
    df1.loc[i, "Artist"] = results["artists"]["items"][0]["name"]
    df1.loc[i, "Followers"] = results["artists"]["items"][0]["followers"]["total"]
    df1.loc[i, "Popularity"] = results["artists"]["items"][0]["popularity"]
    df1.loc[i, "Genre"] = genre
    
    # Print some log
    a = "Genre: " + genre + ", i = " + str(i) + ", n = " + str(n) + " / " + str(results["artists"]["total"]) + ", done! Token n° " + str(t)
    sys.stdout.write('\r'+a)
    
    # Change row
    i += 1
```

Gathering the data is quite long and, as expected, the token expires at some point. The ``spotipy`` module does not yet provide a way to automatically reconnect, so we define a second and pretty straightforward function, ``is_token_expired``, that checks whether the tocken is still valid. The argument comes from the ``credentials`` created before. This is recquired because the Spotify error an expired token generates cannot be used as an exception.

This little and useful function was proposed by user ``ritiek`` on GitHub: https://github.com/plamere/spotipy/issues/209


```python
def is_token_expired(token_info):
    now = int(time.time())
    return token_info['expires_at'] - now < 60
```

Finally, this third function will allow us to keep getting data without throwing an error and stopping the whole process in case the token expires, simply by asking for a new one. We will use it in case of expired token, or if we face another issue that is not related to the Spotify API (connection errors, etc.).


```python
def reconnect():
    
    # These will be used by add_row()
    global token, spotify, t
    
    # Print a log informing that the token expired
    a = "\n Expired token! Generating new token. \n"
    sys.stdout.write('\r' + a)
    
    # Waiting a bit before reconnecting (in case some delay is necessary for the API)
    time.sleep(5)
    
    # Asking for the new token
    token = creds.get_access_token()
    spotify = spotipy.Spotify(token)
    
    # Again a log informing that it worked fine and that the whole process keeps going on
    a = "\n Generated new token, resuming. \n"
    sys.stdout.write('\r' + a)
    
    # Change token number
    t += 1
```

Below is what the whole process looks like.

It is convenient because we have an exhaustive list of genre in Spotify over which to loop, and within each genre we will catch all present artists and their characteristics. The values were accessed just like we did previously, but this time we look at each artist in the genre rather than simply the genre total number of artist: instead of calling artist ``n = 0``, without caring about this particular artist, we will call artists ``n = 0`` up to artist ``n = Total number of artists``, then switch to a new genre.


```python
# Start at row 0, token 1
i = 0
t = 1

# Loop over every genres
for genre in Genres :
            
            # When switching to a new genre, start at artist 0
            n = 0  

            # If the token is expired, reconnect before going further (which otherwise would 
            # stop everything and generate an error we cannot use as an exception)
            if is_token_expired(creds.token_info) :
                
                reconnect()
                
            # A first call is made to grab the total number of artists in this genre
            results = spotify.search(q='genre:' + genre, type='artist', limit = 1, offset = 0)  

            # Loop over every artists in the genre
            for n in range(results["artists"]["total"]):
                
                # This structure allows us to start again at the same artist rather than skiping it
                # in case of an "except" error
                while True:
        
                    # Try to add a new row to the dataframe
                    try:

                        # Before going further, check that the token is still valid, reconnect if
                        # needed
                        if is_token_expired(creds.token_info):
                           
                            reconnect()
                        
                        # Add the new row
                        add_row()
                
                    # These errors might happen because of connection problems or bugs, they are
                    # the only one I encountered and only recquired to try again. Ask for a new token
                    # to make sure.
                    except (ssl.SSLError, ValueError, ConnectionError):
                                                                         
                        print("\n Error, reconnecting. \n")
                        reconnect()
                        
                        continue
        
                    break

# It will take you some time to get there! 
print("Done!")

# Save the file
df1.to_csv("Spotify_Full.csv")
```

As you have noticed there is no log here because I saved the full dataframe in a csv file earlier, and I will not launch this whole thing again (it takes a couple of hours) just for showing some logs. Instead let us have a look at what it looks like in the end.


```python
df1 = pd.read_csv("Spotify_Full.csv", encoding = "ISO-8859-1")
df1.head(10)
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
      <th>Popularity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>pop</td>
      <td>Ed Sheeran</td>
      <td>14003604</td>
      <td>99</td>
    </tr>
    <tr>
      <th>1</th>
      <td>pop</td>
      <td>Sam Smith</td>
      <td>4199743</td>
      <td>97</td>
    </tr>
    <tr>
      <th>2</th>
      <td>pop</td>
      <td>The Weeknd</td>
      <td>8355421</td>
      <td>95</td>
    </tr>
    <tr>
      <th>3</th>
      <td>pop</td>
      <td>Kanye West</td>
      <td>5821775</td>
      <td>93</td>
    </tr>
    <tr>
      <th>4</th>
      <td>pop</td>
      <td>Taylor Swift</td>
      <td>7937350</td>
      <td>95</td>
    </tr>
    <tr>
      <th>5</th>
      <td>pop</td>
      <td>Chris Brown</td>
      <td>5213371</td>
      <td>96</td>
    </tr>
    <tr>
      <th>6</th>
      <td>pop</td>
      <td>Ty Dolla $ign</td>
      <td>1181993</td>
      <td>95</td>
    </tr>
    <tr>
      <th>7</th>
      <td>pop</td>
      <td>Nicki Minaj</td>
      <td>7863550</td>
      <td>95</td>
    </tr>
    <tr>
      <th>8</th>
      <td>pop</td>
      <td>Justin Bieber</td>
      <td>14726943</td>
      <td>97</td>
    </tr>
    <tr>
      <th>9</th>
      <td>pop</td>
      <td>Maroon 5</td>
      <td>9052338</td>
      <td>95</td>
    </tr>
  </tbody>
</table>
</div>




```python
df1.tail(10)
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
      <th>Popularity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>341768</th>
      <td>deep+deep+tech+house</td>
      <td>Fivetone</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>341769</th>
      <td>deep+deep+tech+house</td>
      <td>Nistagmuss</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>341770</th>
      <td>deep+deep+tech+house</td>
      <td>Chris Karpas</td>
      <td>81</td>
      <td>4</td>
    </tr>
    <tr>
      <th>341771</th>
      <td>deep+deep+tech+house</td>
      <td>Gar Doran</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>341772</th>
      <td>deep+deep+tech+house</td>
      <td>Luis Hungria</td>
      <td>25</td>
      <td>1</td>
    </tr>
    <tr>
      <th>341773</th>
      <td>deep+deep+tech+house</td>
      <td>Nick Tonze</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>341774</th>
      <td>deep+deep+tech+house</td>
      <td>Stason Project</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>341775</th>
      <td>deep+deep+tech+house</td>
      <td>Jacopo Ferrari</td>
      <td>16</td>
      <td>4</td>
    </tr>
    <tr>
      <th>341776</th>
      <td>deep+deep+tech+house</td>
      <td>CJ Daedra</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>341777</th>
      <td>deep+deep+tech+house</td>
      <td>Andrew Peret</td>
      <td>11</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



# Summary

In this part we showed:
* how to program a stable function that can search the Spotify API with ``spotipy`` to feed our dataframe for several hours without interrupting;
* how to perform a Kolmogorov-Smirnov test when the parameters are estimated and we test another distribution than the Normal or Exponential (for which there is the Lilliefors test, for instance).

In the next part, we will dive into the last dataframe we created which, as you will see, contains a lot more of information.
