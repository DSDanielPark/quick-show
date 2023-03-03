# Quick-Show
- Quick-Show is a package that allows you to easily and quickly draw plots.
- Quick Show is an abstraction using popular libraries such as sklearn and matplotlib, so it is very light and convenient.
- `Note`: Quick-Show is sub-modules of other packages to manage quickshow more lightly and use more widly. 
This is a project under development as a submodule. With the end of the project, We plan to provide documents in major version 1 and sphinx. It is **NOT** recommended to use prior to major version 1.

<br>

# Install
  ```cmd
  $ pip install quickshow
  ```
<br>
 
# Guide
## 1. Plots related to dimensionality reduction
2D or 3D t-SNE and PCA plots using specific columns of a refined dataframe. 
Create a scatter plot very quickly and easily by inputting a clean dataframe and column names that do not have missing data. 
<br><br>

  `Functions` <br>
    1. `vis_tsne2d` function: Simple visuallization of 2-dimensional t-distributed stochastic neighbor embedding <br>
    2. `vis_tsne3d` function: Simple visuallization of 3-dimensional t-distributed stochastic neighbor embedding <br>
    3. `vis_pca` function: Simple visuallization of Principal Component Analysis (PCA) <br><br>

  <details>
  <summary> See example dataframe... </summary>
  ```
  import pandas as pd
  df = pd.DataFrame([3,2,3,2,3,3,1,1])
  df['val'] = [np.array([np.random.randint(0,10000),np.random.randint(0,10000),np.random.randint(0,10000)]) for x in df[0]]
  df.columns = ['labels', 'values']
  print(df)
  >>>   label   |   values
  >>>   3       |   [8425, 8023, 2019]
  ...
  >>>   1       |   [5551, 8079, 69]
  ```
  </details>

  ```python
  from quickshow import vis_tsne2d, vis_tsne3d, vis_pca

  return_df = vis_tsne2d(df, 'values', 'labels', True, './save/fig1.png')
  return_df = vis_tsne3d(df, 'values', 'labels', True, './savefig2.png')
  return_df = vis_pca(df, 'values', 'labels', 2, True, './savefig3.png')
  return_df = vis_pca(df, 'values', 'labels', 3, True, './savefig4.png')
  ```

  ![](https://github.com/DSDanielPark/quick-show/blob/main/quickshow/output/readme_fig1.png)
  ![](https://github.com/DSDanielPark/quick-show/blob/main/quickshow/output/readme_fig2.png)
  - All function returns the dataframe which used to plot. Thus, use the returned dataframe object to customize your plot. Or use [matplotlib's rcparam](https://matplotlib.org/stable/tutorials/introductory/customizing.html) methods.
  - If the label column does not exist, simply enter `None` as an argument.
  - For more details, please check doc string.
<br>


<br>
<br>


# References
[1] sklearn.manifold.TSNE https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html <br>
[2] sklearn.decomposition.PCA https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html <br>
[3] matplotlib https://matplotlib.org/
<br>

## Contacts
Project Owner(P.O): [Daniel Park, South Korea](https://github.com/DSDanielPark) 
e-mail parkminwoo1991@gmail.com