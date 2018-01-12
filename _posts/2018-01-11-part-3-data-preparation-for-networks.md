---
layout: post
title:  "Part 3"
date:   2018-01-11
excerpt: "Data Preparation for Networks "
image: "/images/part3.PNG"
---

# Part 3: Data Preparation for Networks

In this part, we will prepare an adjacency matrix that will be used in the next part with the package ``networkx``.

The first section creates a symmetric matrix whose elements are the number of common artists between the genres in the rows and columns.

The second section creates a symmetric matrix  whose elements are the sum of the size of the genres in the rows and columns. We should then remove the artists in common as they were counted twice in the outter sum.

The last section divides the first matrix by the second, element-wise. The elements of this matrix are weights representing the share of common artists between any two genres. This division allows for a more meaningfull scale for the weights: two genres sharing 10 artists does not mean the same when the two genres have 100 artists in total and when they have 10000 artists in total.


```python
%reset

import pandas as pd
import numpy as np
import os

os.chdir("C:/Users/antoi/Documents/Spotify_Project")
```

    Once deleted, variables cannot be recovered. Proceed (y/[n])? y
    

## Section 1: creating the matrix of common artists
Let us start by loading the dataset we created in the previous part.


```python
artist = pd.read_csv("Spotify_Artist.csv", encoding = "ISO-8859-1")
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
genres = pd.read_csv("Spotify_Genre.csv", index_col = 0)
genres = genres.sort_values("Size", ascending = False)
genres.to_csv("Ordered_Genres.csv")
genres_index = list(genres.index)
genres_ordered = genres["Genre"]
genres.head()
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
      <th>334</th>
      <td>8.016417</td>
      <td>62.294430</td>
      <td>500544083.0</td>
      <td>5.862069</td>
      <td>757</td>
      <td>dance pop</td>
    </tr>
    <tr>
      <th>976</th>
      <td>7.479576</td>
      <td>60.605067</td>
      <td>228215264.0</td>
      <td>10.031297</td>
      <td>672</td>
      <td>modern rock</td>
    </tr>
    <tr>
      <th>1114</th>
      <td>6.905397</td>
      <td>69.972769</td>
      <td>610646732.0</td>
      <td>5.921331</td>
      <td>661</td>
      <td>pop</td>
    </tr>
    <tr>
      <th>877</th>
      <td>7.086597</td>
      <td>52.987616</td>
      <td>136330099.0</td>
      <td>4.877709</td>
      <td>646</td>
      <td>latin</td>
    </tr>
    <tr>
      <th>1119</th>
      <td>7.185918</td>
      <td>64.167987</td>
      <td>372617074.0</td>
      <td>7.083994</td>
      <td>632</td>
      <td>pop rap</td>
    </tr>
  </tbody>
</table>
</div>



The first step is to group this dataset by artists, and store a list of the genres these artists belong to.


```python
df = pd.DataFrame(artist.groupby("Artist")["Genre"].apply(list))
df.to_csv("Genres_by_Artists.csv")
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



Let us see how we can efficiently use ``numpy``'s vectorization function to detect which genres an artist belongs to within the full list of genres. Recall that the full list of genres is ordered as follows.


```python
genres_ordered.head(10)
```




    334        dance pop
    976      modern rock
    1114             pop
    877            latin
    1119         pop rap
    1208            rock
    779       indie rock
    784     indietronica
    1121        pop rock
    614         folk-pop
    Name: Genre, dtype: object



By doing as below, we create a boolean array, that must be understood like this:
* each "row" is an artist, ordered as in ``df``;
* each "column" is a genre, ordered as in ``genres``.
* For instance, ``True`` on the first row and second column of ``arr`` indicates that artist 1 belongs (among others) to genre 2, that is, the band "!!!" belongs to genre "modern rock", which is ``True``.


```python
# create an array of list of artists' genres
art_genres = np.array(df.Genre)

# initiate a vectorized function that will take list and check if the artists' genres are in the full list of genres
vect = np.vectorize(list.__contains__)

# use the last function with the artists' genres (art_genres) on the full list of genres (genres_ordered)
arr = vect(art_genres[:, None], genres_ordered)
arr
```

    array([[False,  True, False, ..., False, False, False],
           [False, False, False, ..., False, False, False],
           [False, False, False, ..., False, False, False],
           ..., 
           [False, False, False, ..., False, False, False],
           [False, False, False, ..., False, False, False],
           [False, False, False, ..., False, False, False]], dtype=bool)

This is really convenient because it uses boolean data which makes computations much faster. Let us see how to get to our matrix now, by subsetting the previous array then summing along the columns.

We start by only using the 3rd column (``t[:,2]``) for illustration, so we subset the previous ndarray to only show rows where there are ``True`` in this column (notice the 3rd column of ``True`` only).


```python
print(arr[arr[:,2]].shape)
arr[arr[:,2]]
```

    (661, 1520)
    array([[ True, False,  True, ..., False, False, False],
           [ True, False,  True, ..., False, False, False],
           [ True,  True,  True, ..., False, False, False],
           ..., 
           [False, False,  True, ..., False, False, False],
           [ True, False,  True, ..., False, False, False],
           [ True, False,  True, ..., False, False, False]], dtype=bool)



We can now sum along the columns to get the number of artists (the number of ``True``) per column. Since we did this on a subset of the full array that contains only the artists that belong to genre 3 (3rd column), these sums on the columns represent the number of artists that also belong to another genre, which is represented by their row. 

For instance, below, we can read that 370 artists belong to the 3rd genre (since we used it to subset the array) and to the 1st genre (1st column). Notice that the third column shows 661, which is the number of row of the subset of ``arr`` for the 3rd genre as can be seen above.


```python
np.sum(arr[arr[:, 2]], axis = 0)
```




    array([370,  47, 661, ...,   0,   0,   0])



We can now do this for the whole list (not just column 3) of genres using a for loop, and we obtain the following.


```python
# subset t for rows where column i is always True (ie all artists that belong to genre i), then sum for each column
ndarr = np.array([np.sum(arr[arr[:, i]], axis = 0) for i in range(len(genres_ordered))])
ndarr
```




    array([[754,   5, 370, ...,   0,   0,   0],
           [  5, 671,  47, ...,   0,   0,   0],
           [370,  47, 661, ...,   0,   0,   0],
           ..., 
           [  0,   0,   0, ...,   1,   0,   0],
           [  0,   0,   0, ...,   0,  38,   0],
           [  0,   0,   0, ...,   0,   0,  19]])



**This whole process could have been done with a nested ``for`` loop which would have been clearer, but the gain in computation time is massive by doing as we did (a few seconds versus more than 30 minutes on my computer)**

This newly created array can be used as a dataframe.


```python
df = pd.DataFrame(ndarr, columns = genres_ordered, index = genres_ordered)
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
      <th>Genre</th>
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
      <td>754</td>
      <td>5</td>
      <td>370</td>
      <td>13</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>modern rock</th>
      <td>5</td>
      <td>671</td>
      <td>47</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>pop</th>
      <td>370</td>
      <td>47</td>
      <td>661</td>
      <td>4</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>latin</th>
      <td>13</td>
      <td>0</td>
      <td>4</td>
      <td>646</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>pop rap</th>
      <td>158</td>
      <td>6</td>
      <td>120</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


We can thus see, for instance, that "dance pop" and "pop" genres share 370 artists, and that "modern rock" and "indie rock" share 393 artists, which makes sense.

## Section 2: creating the matrix of sizes
Here we see the genres and their respective sizes.


```python
sum_genres_ordered = pd.DataFrame({"Genre":genres_ordered, "Size":genres["Size"]})
sum_genres_ordered.index = range(sum_genres_ordered.shape[0])
sum_genres_ordered.head()
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
      <th>Size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>dance pop</td>
      <td>757</td>
    </tr>
    <tr>
      <th>1</th>
      <td>modern rock</td>
      <td>672</td>
    </tr>
    <tr>
      <th>2</th>
      <td>pop</td>
      <td>661</td>
    </tr>
    <tr>
      <th>3</th>
      <td>latin</td>
      <td>646</td>
    </tr>
    <tr>
      <th>4</th>
      <td>pop rap</td>
      <td>632</td>
    </tr>
  </tbody>
</table>
</div>



We can first use an outer sum on the dataframe above to obtain the following.


```python
sums_df = pd.DataFrame(np.add.outer(sum_genres_ordered['Size'], sum_genres_ordered['Size']),
                       columns = sum_genres_ordered['Genre'].values,
                       index = sum_genres_ordered['Genre'].values)
sums_df.head()
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
  </thead>
  <tbody>
    <tr>
      <th>dance pop</th>
      <td>1514</td>
      <td>1429</td>
      <td>1418</td>
      <td>1403</td>
      <td>...</td>
      <td>801</td>
      <td>797</td>
      <td>795</td>
      <td>776</td>
    </tr>
    <tr>
      <th>modern rock</th>
      <td>1429</td>
      <td>1344</td>
      <td>1333</td>
      <td>1318</td>
      <td>...</td>
      <td>716</td>
      <td>712</td>
      <td>710</td>
      <td>691</td>
    </tr>
    <tr>
      <th>pop</th>
      <td>1418</td>
      <td>1333</td>
      <td>1322</td>
      <td>1307</td>
      <td>...</td>
      <td>705</td>
      <td>701</td>
      <td>699</td>
      <td>680</td>
    </tr>
    <tr>
      <th>latin</th>
      <td>1403</td>
      <td>1318</td>
      <td>1307</td>
      <td>1292</td>
      <td>...</td>
      <td>690</td>
      <td>686</td>
      <td>684</td>
      <td>665</td>
    </tr>
    <tr>
      <th>pop rap</th>
      <td>1389</td>
      <td>1304</td>
      <td>1293</td>
      <td>1278</td>
      <td>...</td>
      <td>676</td>
      <td>672</td>
      <td>670</td>
      <td>651</td>
    </tr>
  </tbody>
</table>
</div>



However, we showed in the previous section that genres had artists in common, so we must remove them because they where counted twice in the above computation. To do so, we simply subtract the previous matrix from this one, element-wise.


```python
sums_df = sums_df.subtract(df)
sums_df.head()
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
  </thead>
  <tbody>
    <tr>
      <th>dance pop</th>
      <td>760</td>
      <td>1424</td>
      <td>1048</td>
      <td>1390</td>
      <td>...</td>
      <td>801</td>
      <td>797</td>
      <td>795</td>
      <td>776</td>
    </tr>
    <tr>
      <th>modern rock</th>
      <td>1424</td>
      <td>673</td>
      <td>1286</td>
      <td>1318</td>
      <td>...</td>
      <td>716</td>
      <td>712</td>
      <td>710</td>
      <td>691</td>
    </tr>
    <tr>
      <th>pop</th>
      <td>1048</td>
      <td>1286</td>
      <td>661</td>
      <td>1303</td>
      <td>...</td>
      <td>705</td>
      <td>701</td>
      <td>699</td>
      <td>680</td>
    </tr>
    <tr>
      <th>latin</th>
      <td>1390</td>
      <td>1318</td>
      <td>1303</td>
      <td>646</td>
      <td>...</td>
      <td>690</td>
      <td>686</td>
      <td>684</td>
      <td>665</td>
    </tr>
    <tr>
      <th>pop rap</th>
      <td>1231</td>
      <td>1298</td>
      <td>1173</td>
      <td>1277</td>
      <td>...</td>
      <td>676</td>
      <td>672</td>
      <td>670</td>
      <td>651</td>
    </tr>
  </tbody>
</table>
</div>



## Section 3: creating the adjacency matrix
Finally, we can divide the first matrix by the second, element-wise, to obtain the percentage of common artists. We also set the diagonal to zero, otherwise the network will read that all genres have loops. 


```python
# Creates shares of common artists rather than counts of common artists
adj_df = df.divide(sums_df) * 100
np.fill_diagonal(adj_df.values, 0)
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
      <th>Genre</th>
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
      <td>0.351124</td>
      <td>35.305344</td>
      <td>0.935252</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>modern rock</th>
      <td>0.351124</td>
      <td>0.000000</td>
      <td>3.654743</td>
      <td>0.000000</td>
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
      <td>0.306984</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>latin</th>
      <td>0.935252</td>
      <td>0.000000</td>
      <td>0.306984</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>pop rap</th>
      <td>12.835093</td>
      <td>0.462250</td>
      <td>10.230179</td>
      <td>0.078309</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>


Finally, we can save the dataframe to load it in the next Part.


```python
adj_df.to_csv("Adjacency_Matrix.csv")
```
