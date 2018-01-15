---
layout: post
title:  "Part 6"
date:   2018-01-11
excerpt: "Machine Learning: Gradient Boosting and Neural Networks "
image: "/images/Posts_Images/Part6/part6.png"
---
<span class="image fit"><img src="{{ "/images/Posts_Images/Part6/part6.png" | absolute_url }}" alt="" /></span>


In this part (and the next one), we will try to predict the popularity of an artist based solely on its combination of genres.

We will first transform the data to be suitable for the model.

Then, we will use the ``lightgbm`` (Light Gradient Boosting Machine) module to try and build a model for prediction. ``lightgbm`` is developed by Microsoft, and is similar to the well known module ``xgboost``. It is at least as accurate, but almost everytime faster. See this link to learn more about the package: https://github.com/Microsoft/LightGBM . I also found ``lightbm`` much easier to use, from the installation to the model building, compared to ``xgboost``. 

Finally, we will us ``keras`` to build a Neural Network and try to improve our predictions.

## Setting up


```python
%reset

import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import os
import sys
import graphviz
import warnings
import time

from matplotlib import style

from sklearn.preprocessing import StandardScaler, Binarizer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn import decomposition

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.optimizers import SGD
from keras import regularizers
from keras.wrappers.scikit_learn import KerasRegressor

from IPython.display import clear_output

warnings.filterwarnings('ignore')

os.environ["PATH"] += os.pathsep + "C:/Program Files (x86)/graphviz-2.38/release/bin"

os.chdir("C:/Users/antoi/Documents/Spotify_Project")

style.use('white')
```

    Once deleted, variables cannot be recovered. Proceed (y/[n])? y
    

    Using TensorFlow backend.
    

## Importing and preparing the data
What comes next is very similar to what we already did before, and you will certainly recognize the dataframe ``artist_genres`` from before.


```python
artists = pd.read_csv("Spotify_Artist.csv", encoding = "ISO-8859-1")
artists_genres = pd.DataFrame(artists.groupby("Artist")["Genre"].apply(list))
artists_genres["Artist"] = artists_genres.index
artists_genres.index = range(len(artists_genres))
artists_genres.head()
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
      <td>[modern rock, indie pop, indietronica, indie r...</td>
      <td>!!!</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[nu jazz, electro swing, balkan brass]</td>
      <td>!Dela Dap</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[futurepop, neo-synthpop]</td>
      <td>!Distain</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[bass trap, electronic trap]</td>
      <td>!PVNDEMIK</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[technical death metal]</td>
      <td>!T.O.O.H.!</td>
    </tr>
  </tbody>
</table>
</div>



Now, what would be usefull is to obtain binary variables, where each variable is a genre. If the artist belongs to the genre, its value in the dataframe will be 1, otherwise it will be 0. Since it can belong to many genres, ``scikit-learn``'s ``OneHotEncode`` function is not suitable. There exists the function ``MultiLabelBinarizer``, but it is used in pipelines for the output variable in multi-classification problems. We will just do it ourselves! It actually takes one line.


```python
# "One hot encoding" for several genres
ml_df = pd.DataFrame({k: 1 for k in x} for x in artists_genres.Genre).fillna(0).astype(int)

# Adding the name of the artist, used next for merging
ml_df = pd.concat([artists_genres["Artist"], ml_df] , axis=1)

ml_df.head()
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
      <th>a cappella</th>
      <th>abstract</th>
      <th>abstract beats</th>
      <th>...</th>
      <th>zouglou</th>
      <th>zouk</th>
      <th>zouk riddim</th>
      <th>zydeco</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>!!!</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>!Dela Dap</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>!Distain</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>!PVNDEMIK</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>!T.O.O.H.!</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



To ensure you that the above dataset is not empty (as we can only see zeros in the preview), we can see, by taking the sum for the columns, that we have 466 artists in "indie psych-rock", and 181 in "chicago house", which means we do have 1s in our dataset.


```python
sum(ml_df["indie psych-rock"])
```




    466




```python
sum(ml_df["chicago house"])
```




    181



Now, from a dataset we created in a previous part, let us import the popularity values for the artists...


```python
artists_values = pd.read_csv("Artists_Values.csv", encoding = "ISO-8859-1")
del artists_values["Artist.1"]
artists_values.head()
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
      <th>Followers</th>
      <th>Popularity</th>
      <th>Variety</th>
      <th>log Followers</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Wire</td>
      <td>47637.0</td>
      <td>55</td>
      <td>43</td>
      <td>10.771365</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Broadcast</td>
      <td>40622.0</td>
      <td>55</td>
      <td>40</td>
      <td>10.612065</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Grouper</td>
      <td>42968.0</td>
      <td>55</td>
      <td>39</td>
      <td>10.668211</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Dr. John</td>
      <td>84367.0</td>
      <td>60</td>
      <td>38</td>
      <td>11.342932</td>
    </tr>
    <tr>
      <th>4</th>
      <td>The Feelies</td>
      <td>19065.0</td>
      <td>48</td>
      <td>36</td>
      <td>9.855609</td>
    </tr>
  </tbody>
</table>
</div>



... and add them to ``ml_df``. We used merging based on the artists names.  Now our last column is the popularity.


```python
ml_df = ml_df.merge(artists_values[["Popularity", "Artist"]] , on = "Artist")
del ml_df["Artist"]
ml_df.head()
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
      <th>a cappella</th>
      <th>abstract</th>
      <th>abstract beats</th>
      <th>abstract hip hop</th>
      <th>...</th>
      <th>zouk</th>
      <th>zouk riddim</th>
      <th>zydeco</th>
      <th>Popularity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>58</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>37</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>16</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



We can save the dataset.


```python
ml_df.to_csv("Deep_Learning_Spotify.csv")
```

# Section 1: Building the model with lightgbm
We are now ready to separate our data between training, evaluation and test sets. We first import the dataset we saved above, and shuffle the data to make sure no pattern exists in each set.


```python
df = pd.read_csv("Deep_Learning_Spotify.csv", index_col = 0)
df = df.sample(frac=1).reset_index(drop=True)
```

We extract the values in our dataset, then use Principal Component Analysis for dimensionality reduction and to avoid overfitting. This allowed for a significant gain in accuracy and greater speed in training. After that, we separate our variables in two sets, ``training`` and ``test``, because we will use Grid Search Cross Validation for hyper-parameter tuning, which already splits the training set into training and evaluation. However, after we identified the best parameters with Grid Search CV, we will train the model further with these parameters, and we will need ``training`` to be separated between ``train`` and ``eval``. Since we will use Poisson regression, as popularity values are non-negative integers, we do not need to scale the data. There is, however, code commented out that could be used to do so.


```python
# Extract the values in the dataframe
X, y = df.iloc[:,:-1], df.iloc[:,-1]

# Use Principal Component Analysis
pca = decomposition.PCA(n_components = 100)
X = pca.fit_transform(X)

# Separate into training and test
X_training, X_test, y_training, y_test = train_test_split(X, y, test_size = 0.25, random_state = 33)

# Separate into training and evaluation (in the end we thus have train: 50%, eval: 25%, test: 25%)
X_train, X_eval, y_train, y_eval = train_test_split(X_training, y_training, test_size = 0.33, random_state = 33)
```

We can now create our estimator. We will use GridSearchCV on the number of leaves and maximum number of bins only, and 3-fold cross validation only (the default number), because the search gets too long otherwise on my laptop, but the method is exactly the same. The other parameters were tuned by observing the plots we will present below. 

We will be using Poisson as objective. Since we are not doing inference but prediction instead, and that our sample size is large, we do not need to care about the variance being equal to the mean. 


```python
# create the estimator, which is a regressor that with poisson objective
# bagging_fraction, feature_fraction, and bagging_freq are used to prevent over-fitting.
estimator = lgb.LGBMRegressor(objective = "poisson", learning_rate = 0.01, bagging_fraction = 0.3, feature_fraction = 0.3,
                              bagging_freq = 6, num_boost_round = 1000)

# create the parameter grid for the grid search
param_grid = {"num_leaves": [5, 10], "max_bin": [5, 10]}

# create the grid search and cross validation object, using the estimator and parameter grid defined above
gbm = GridSearchCV(estimator, param_grid, scoring = "neg_mean_squared_error", verbose = 1)

# perform the grid search
gbm.fit(X_training, y_training)

# print out the best parameters
print('Best parameters found by grid search:', gbm.best_params_)
```

    Fitting 3 folds for each of 4 candidates, totalling 12 fits
    

    [Parallel(n_jobs=1)]: Done  12 out of  12 | elapsed:  3.6min finished
    

    Best parameters found by grid search: {'max_bin': 10, 'num_leaves': 10}
    

We can have a look at the different models that were fit and their scores.


```python
pd.DataFrame(gbm.cv_results_)
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
      <th>a cappella</th>
      <th>abstract</th>
      <th>abstract beats</th>
      <th>abstract hip hop</th>
      <th>...</th>
      <th>zouglou</th>
      <th>zouk</th>
      <th>zouk riddim</th>
      <th>zydeco</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.001142</td>
      <td>0.000469</td>
      <td>0.000884</td>
      <td>0.001151</td>
      <td>...</td>
      <td>0.001372</td>
      <td>0.001545</td>
      <td>0.001375</td>
      <td>0.000499</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.000250</td>
      <td>0.000403</td>
      <td>0.000058</td>
      <td>0.000341</td>
      <td>...</td>
      <td>0.000280</td>
      <td>0.000314</td>
      <td>0.000277</td>
      <td>0.001251</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.000781</td>
      <td>0.000686</td>
      <td>0.000440</td>
      <td>0.005809</td>
      <td>...</td>
      <td>0.001095</td>
      <td>0.001214</td>
      <td>0.001083</td>
      <td>0.001192</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.000205</td>
      <td>0.000559</td>
      <td>0.000619</td>
      <td>0.001933</td>
      <td>...</td>
      <td>0.000299</td>
      <td>0.000331</td>
      <td>0.000300</td>
      <td>0.001232</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.000057</td>
      <td>0.000077</td>
      <td>0.000039</td>
      <td>0.004576</td>
      <td>...</td>
      <td>0.000101</td>
      <td>0.000115</td>
      <td>0.000101</td>
      <td>0.000322</td>
    </tr>
  </tbody>
</table>
</div>



We will keep the parameters from the model that ranked first for max_bin and num_leaves, and keep the same parameters for the others. This time we will allow the model to train for a longer time to achieve better accuracy. Using early stopping, we will prevent the model to train more than it should and start overfitting (the evaluation error would increase).


```python
params = gbm.best_params_

params["objective"] = "poisson"
params["metric"] = {"rmse", "l1"}
params["learning_rate"] = 0.1
params["bagging_fraction"] = 0.01
params["bagging_freq"] = 20
params["feature_fraction"] = 0.01
params["num_boost_round"] = 20000
params["early_stopping_rounds"] = 200
```

We can convert our data to the ``lightgbm`` "Dataset" format for greater speed (this is not accepted by ``GridSearchCV``). We will store metrics in ``evals_result`` to further analyze them. Then we train the model with the parameters we defined above, including those from grid search. This is where it was useful to further split the data. Notice that the test set has not been and will not be seen in any way by the model. 


```python
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_eval, y_eval, reference = lgb_train)

evals_result = {}

model = lgb.train(params,
                  lgb_train,
                  valid_sets = [lgb_eval, lgb_train],
                  valid_names = ["validation", "training"],
                  verbose_eval = 100,
                  evals_result = evals_result)
```

    Training until validation scores don't improve for 200 rounds.
    [100]	training's l1: 10.9251	training's rmse: 13.664	validation's l1: 10.8979	validation's rmse: 13.6235
    [200]	training's l1: 10.7184	training's rmse: 13.3822	validation's l1: 10.6974	validation's rmse: 13.3406
    [300]	training's l1: 10.4327	training's rmse: 13.1272	validation's l1: 10.4087	validation's rmse: 13.0857
    [400]	training's l1: 10.221	training's rmse: 12.9655	validation's l1: 10.2026	validation's rmse: 12.9406
    [500]	training's l1: 10.1951	training's rmse: 12.8659	validation's l1: 10.1867	validation's rmse: 12.8523
    [600]	training's l1: 10.0871	training's rmse: 12.7932	validation's l1: 10.0761	validation's rmse: 12.7735
    [700]	training's l1: 10.0594	training's rmse: 12.7116	validation's l1: 10.0492	validation's rmse: 12.6988
    [800]	training's l1: 10.0151	training's rmse: 12.6853	validation's l1: 10.0048	validation's rmse: 12.6737
    [900]	training's l1: 9.94153	training's rmse: 12.6476	validation's l1: 9.93703	validation's rmse: 12.6404
    [1000]	training's l1: 9.8814	training's rmse: 12.6246	validation's l1: 9.88113	validation's rmse: 12.614
    [1100]	training's l1: 9.86952	training's rmse: 12.5833	validation's l1: 9.87773	validation's rmse: 12.5903
    [1200]	training's l1: 9.85737	training's rmse: 12.5788	validation's l1: 9.87131	validation's rmse: 12.5934
    [1300]	training's l1: 9.83257	training's rmse: 12.5839	validation's l1: 9.84123	validation's rmse: 12.5831
    [1400]	training's l1: 9.7807	training's rmse: 12.5571	validation's l1: 9.79276	validation's rmse: 12.5618
    [1500]	training's l1: 9.84934	training's rmse: 12.5867	validation's l1: 9.87195	validation's rmse: 12.6099
    [1600]	training's l1: 9.87087	training's rmse: 12.5442	validation's l1: 9.89071	validation's rmse: 12.5574
    Early stopping, best iteration is:
    [1480]	training's l1: 9.747	training's rmse: 12.5651	validation's l1: 9.75751	validation's rmse: 12.5663
    

Our validation error follows very closely our training error, which is very good.


```python
%matplotlib inline
ax = lgb.plot_metric(evals_result, metric='rmse')
plt.show()
```


![metric]({{ "./images/Posts_Images/Part6/part6-1.png" | absolute_url }})

It is possible to have a look at the most important features to determine the popularity of an artist. Since we used principal component analysis, this is not very informative. 


```python
%matplotlib inline
import matplotlib.pyplot as plt
lgb.plot_importance(model, max_num_features=10)
plt.show()
```

![component]({{ "./images/Posts_Images/Part6/part6-2.png" | absolute_url }})

It is however possible to inspect the components of the pca and how much they are informed by our original features.


```python
pca_df = pd.DataFrame(np.abs(pca.components_),columns=df.columns[:(-1)])
pca_df.head()
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
<table border="1" class="dataframe" style="max-width:300px;">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>a cappella</th>
      <th>abstract</th>
      <th>abstract beats</th>
      <th>abstract hip hop</th>
      <th>abstract idm</th>
      <th>abstractro</th>
      <th>accordeon</th>
      <th>accordion</th>
      <th>acid house</th>
      <th>acid jazz</th>
      <th>...</th>
      <th>yugoslav rock</th>
      <th>zapstep</th>
      <th>zeuhl</th>
      <th>zillertal</th>
      <th>zim</th>
      <th>zolo</th>
      <th>zouglou</th>
      <th>zouk</th>
      <th>zouk riddim</th>
      <th>zydeco</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.001142</td>
      <td>0.000469</td>
      <td>0.000884</td>
      <td>0.001151</td>
      <td>0.000862</td>
      <td>0.000186</td>
      <td>0.000575</td>
      <td>0.000914</td>
      <td>0.000287</td>
      <td>0.000050</td>
      <td>...</td>
      <td>0.001245</td>
      <td>0.002394</td>
      <td>0.000655</td>
      <td>0.001404</td>
      <td>0.000527</td>
      <td>0.007161</td>
      <td>0.001372</td>
      <td>0.001545</td>
      <td>0.001375</td>
      <td>0.000499</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.000250</td>
      <td>0.000403</td>
      <td>0.000058</td>
      <td>0.000341</td>
      <td>0.000088</td>
      <td>0.000421</td>
      <td>0.000116</td>
      <td>0.000186</td>
      <td>0.002098</td>
      <td>0.000334</td>
      <td>...</td>
      <td>0.000252</td>
      <td>0.000248</td>
      <td>0.000023</td>
      <td>0.000285</td>
      <td>0.000112</td>
      <td>0.003551</td>
      <td>0.000280</td>
      <td>0.000314</td>
      <td>0.000277</td>
      <td>0.001251</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.000781</td>
      <td>0.000686</td>
      <td>0.000440</td>
      <td>0.005809</td>
      <td>0.000724</td>
      <td>0.000409</td>
      <td>0.000435</td>
      <td>0.000696</td>
      <td>0.000214</td>
      <td>0.001207</td>
      <td>...</td>
      <td>0.000963</td>
      <td>0.001118</td>
      <td>0.000807</td>
      <td>0.001090</td>
      <td>0.000400</td>
      <td>0.000439</td>
      <td>0.001095</td>
      <td>0.001214</td>
      <td>0.001083</td>
      <td>0.001192</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.000205</td>
      <td>0.000559</td>
      <td>0.000619</td>
      <td>0.001933</td>
      <td>0.000264</td>
      <td>0.000503</td>
      <td>0.000118</td>
      <td>0.000188</td>
      <td>0.002079</td>
      <td>0.001241</td>
      <td>...</td>
      <td>0.000261</td>
      <td>0.001542</td>
      <td>0.000355</td>
      <td>0.000295</td>
      <td>0.000108</td>
      <td>0.009686</td>
      <td>0.000300</td>
      <td>0.000331</td>
      <td>0.000300</td>
      <td>0.001232</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.000057</td>
      <td>0.000077</td>
      <td>0.000039</td>
      <td>0.004576</td>
      <td>0.000039</td>
      <td>0.000063</td>
      <td>0.000040</td>
      <td>0.000057</td>
      <td>0.001073</td>
      <td>0.008057</td>
      <td>...</td>
      <td>0.000091</td>
      <td>0.000113</td>
      <td>0.000242</td>
      <td>0.000101</td>
      <td>0.000010</td>
      <td>0.002303</td>
      <td>0.000101</td>
      <td>0.000115</td>
      <td>0.000101</td>
      <td>0.000322</td>
    </tr>
  </tbody>
</table>
</div>



## Predicting popularities
Now that our model is fitted, we can use it to perform predictions on the test set.


```python
y_pred = model.predict(X_test)
y_pred = np.round(y_pred, 0)
```

We have 5 predictions that go above 100, so we assign them the value of 100, then print the Root Mean Squared Errors and Mean Absolute Errors metrics for further comparison.  


```python
print(str(np.sum(y_pred > 100)) + " predictions above 100")
y_pred[y_pred > 100] = 100

print('The rmse of prediction is:', np.round(mean_squared_error(y_test, y_pred) ** 0.5, 2))
print("The mae of prediction is:", np.round(np.abs(y_test - y_pred).mean(), 2))
```

    0 predictions above 100
    The rmse of prediction is: 12.64
    The mae of prediction is: 9.82
    

## Is it better than random ? 
First we can compare our predictions with purely random draws from the interval $$[0, 100]$$. We can see that, on average, there is an absolute error of around 40 between the "prediction" and the actual value.


```python
rand_pred1 = np.random.random_integers(0, 101, y_test.shape[0])
print('The rmse of prediction is:', np.round(mean_squared_error(y_test, rand_pred1) ** 0.5, 2))
print("The mae of prediction is:", np.round(np.abs(y_test - rand_pred1).mean(), 2))
```

    The rmse of prediction is: 46.93
    The mae of prediction is: 38.58
    

Then, we can compare our predictions with random draws from the empirical distribution of popularity values. This approximately divides both errors by 2.


```python
range_to_pick_from = pd.Series(y_training).value_counts().index.tolist()
number_of_picks = y_test.shape[0]
prob = pd.Series(y_training).value_counts().values / sum(pd.Series(y_training).value_counts().values)
rand_pred2 = np.random.choice(range_to_pick_from, number_of_picks, p = prob)
# pred = scaler_Y.inverse_transform(pred)
print('The rmse of prediction is:', np.round(mean_squared_error(y_test, rand_pred2) ** 0.5, 2))
print("The mae of prediction is:", np.round(np.abs(y_test - rand_pred2).mean(), 2))
```

    The rmse of prediction is: 25.76
    The mae of prediction is: 20.0
    

In the end our model divides the errors, by approximately 2 again. It thus performs 4 times better than purely random draws and 2 times better than random draws from the empirical distribution. That is not so bad with only genres belonging as features!

However, one last check to make before leaving it as is: we should plot our predictions against our real values.


```python
%matplotlib inline
import seaborn as sns

pred_df = pd.DataFrame({"Prediction": y_pred, "Test": y_test})
import seaborn as sns
sns.lmplot("Prediction", "Test", data = pred_df, size = 10, aspect = 2, scatter_kws = {"alpha" : 0.1})
plt.show()
```


<span class="image fit"><img src="{{ "/images/Posts_Images/Part6/part6-3.png" | absolute_url }}" alt="" /></span>


The regression line between the actual and predicted values has correct coefficients (approximately 0 as intercept and 1 as slope). But there is a lot of noise in the predictions, certainly too much.

Nothing improved this rather poor performance, from adding more or less components to using ``regression_l2`` as objective function or many different configurations for our tree model.

Even though the model was correctly specified, and we did what was necessary to prevent over-fitting while maximizing the accuracy, another model might be more adapted. 

And from what our data looks like, it might be Neural Networks. 

It is often interesting to try something else before resorting to Neural Networks, as we did, but in our case it might be our solution. 

# Building the model with keras

## Preparing the data

We can reload the data since we won't be using Principal Component Analysis here. We will rescale the popularity variable for better accuracy since we will not be using Poisson objective because it does not perform as well here. 


```python
df = pd.read_csv("Deep_Learning_Spotify.csv", index_col = 0)
df = df.sample(frac = 1).reset_index(drop = True)

dataset = df.values
X = dataset[:,0:(-1)]
Y = dataset[:,(-1)]

# Training data + validation data (X_fit, y_fit): 75%, test data: 25%
X_training, X_test, y_training, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 1)
del X, Y, dataset

# scaler_X = StandardScaler().fit(X_fit)
scaler_Y = StandardScaler().fit(y_training.reshape(-1, 1))
y_training = scaler_Y.transform(y_training.reshape(-1, 1))
```

Let us continue by defining two callbacks that will be very convenient for when building our neural network. They allow for live plotting of the loss metrics during training. They were nicely proposed by users ``stared`` and ``kav`` on GitHub and can be found here: https://gist.github.com/stared/dfb4dfaf6d9a8501cd1cc8b8cb806d2e


```python
class PlotLosses(Callback):
    
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.fig = plt.figure()
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.show();
        
plot_losses = PlotLosses()
```

## Building the model

Because neural networks are much more computationaly intensive than the models in ``lightgbm`` for instance, so we will not perform grid search and cross validation as we did before (I am doing this with a dualcore CPU, 8Gb of RAM, and I don't have a dedicated GPU, which is quite limited).


We use two other callbacks, ``EarlyStopping`` and ``ModelCheckpoint``. The first one is self explanatory, and the second saves the model with the best performance so far during the training.


```python
# Set seed for reproducibility
seed = 42
np.random.seed(seed)

# Define callbacks
stop = EarlyStopping(monitor = "val_loss", min_delta = 0.00001, patience = 5)
check = ModelCheckpoint(filepath="best_model.hdf5", verbose = 1, save_best_only = True) 
```

We can then build the Sequential model as follows. 

We will use one hidden layer, with as many neurons as there are features. The hidden layer has a uniform kernel initializer, and Rectified Linear Units as activation function (ReLu) The output layer has a linear activation function on a single neuron because we are dealing with a regression problem.

Additionaly, to prevent over-fitting we use dropouts between the first and second layers, and between the second and third layers. It is enough and more efficient in our case than when using kernel regularizers that lead to under-fitting.


```python
# Build the model, no kerner regularizer, dropout is enough, otherwise underfitting
mdl1 = Sequential()
mdl1.add(Dropout(0.1, input_shape=(X_training.shape[1],)))
mdl1.add(Dense(X_training.shape[1] , input_dim = X_training.shape[1], kernel_initializer = "uniform", activation = "relu"))
mdl1.add(Dropout(0.2))
mdl1.add(Dense(1, init = "uniform", activation = "linear"))
```

We can now compile the model, using Stochastic Gradient Descent with Nesterov accelerated gradient descent method and momentum as optimizer. Adam optimizer is often used but better results were achieved using this. 


```python
# Compile the model, using stochastic gradient descent with nesterov momentum (worked better than adam optimizer), no decay
# Minimize the mean absolute error rather than the mean squared error when possible
mdl1.compile(loss = "mse", optimizer = SGD(lr = 0.01, momentum = 0.9, nesterov = True), metrics = ["mae"])
```

Finally, the model is ready to be fitted to the training data. For speed and to prevent over-fitting, the batch is rather large. We use the callbacks defined earlier.


```python
# Fit the model, train on 50% of full dataset, validate on 25% of full dataset -> validate = 1/3 of X_fit and y_fit
mdl1.fit(X_training, y_training, validation_split = 0.33, epochs = 100, batch_size = 512,
         verbose = 1, callbacks = [check, plot_losses] )
```


<span class="image fit"><img src="{{ "/images/Posts_Images/Part6/part6-4.png" | absolute_url }}" alt="" /></span>


    94697/94697 [==============================] - 43s - loss: 0.3513 - mean_absolute_error: 0.4502 - val_loss: 0.3508 - val_mean_absolute_error: 0.4489


The plot might seem unusual, with the validation loss below the training loss, but it is actually due to the dropouts we introduced: they are used for training but not for evaluation, so the validation loss can be higher (see this: https://keras.io/getting-started/faq/).

Our validation and training errors converge around the last epoch, which is were we will stop training. Indeed, when the validation error starts going above the training error, the model is learning patterns specific to the training set: it is over-fitting.

It is considered good practice to save and reload your model before going further. Since it was automatically saved for the best epoch thanks to our callbacks, we just have to load it.


```python
mdl = load_model("best_model.hdf5")
```

## Predictions
Hopefully we will achieve better results this time. We perform our predictions on the test set, data that was not seen in any way by the model during training, then transform the predictions back to their original scale using the scaler we created at the beginning of this section.


```python
y_pred = mdl.predict(X_test)
y_pred = scaler_Y.inverse_transform(y_pred).flatten()
y_pred = np.round(y_pred, 0)
```

We can now compute the RMSE and MAE for our predictions.


```python
rmse = np.sqrt(np.mean((y_pred - y_test)**2))
print("rmse = " + str(np.round(rmse, 2)))
mae = np.mean(abs(y_pred - y_test))
print("mae = " + str(np.round(mae, 2)))
```

    rmse = 10.71
    mae = 7.75
    

There is a nice improvement from the predictions with the ``lightgbm`` model, but let us plot them before rejoincing.


```python
pred_df = pd.DataFrame({"Prediction": y_pred, "Test": y_test})
import seaborn as sns
sns.lmplot("Prediction", "Test", data = pred_df, size = 10, aspect = 2, scatter_kws = {"alpha" : 0.1})
plt.show()
```


<span class="image fit"><img src="{{ "/images/Posts_Images/Part6/part6-5.png" | absolute_url }}" alt="" /></span>


**That is much better!** The linear relationship is much clearer. There is still some noise, but the predictions are much more concentrated around the $$y = x$$ axis. Again, the intercept is close to 0 and the slope to 1.

## Conclusion

By simply knowing the combination of genres an artist belongs to, we were able to predict its popularity with an average error of 7.7 and, compared to the previous section, these predictions are more accurate!

The model seems to have more noise for low values, but for a small model like the one we built it is very encouraging, particularly after the relative failure of the previous model. Imagine what we could do with more computing power and proper hyper-parameter tuning!

# Summary

In this part:
* We quickly and efficiently encoded our dataset to right the format;
* We built a tree model using ``lightgbm`` and grid search cross-validation, to try to predict artists' popularity values;
* To improve our predictions, we built a neural network using ``keras``.

Thats's it for this serie! I might try to do another one using song level data in the future. There are interesting features that can be used, which can be found here: https://developer.spotify.com/web-api/get-audio-features/ 

## Bonus: a function to predict popularity on Spotify



```python
df = pd.read_csv("Deep_Learning_Spotify.csv", index_col = 0)

def prediction(list_of_genres) :

    ranks = []
    pred = [0] * 1520
    
    for g in list_of_genres: 
        ranks.append(df.columns.get_loc(g))

    for r in ranks:
        pred[r] = 1

    preds = scaler_Y.inverse_transform(mdl.predict(np.array(pred).reshape(1,-1)))
    
    print("The predicted Spotify popularity for this artist is " + str(int(round(preds[0][0], 0))))
```

You have the project of starting a throat-singing-shoegaze band ? 


```python
prediction(["throat-singing", "shoegaze"])
```

    The predicted Spotify popularity for this artist is 7
    

You might want to think twice before rushing into it (or not, that is what music is all about)!
