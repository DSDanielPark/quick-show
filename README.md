# Quick-Show

[![Contributor Covenant](https://img.shields.io/badge/contributor%20covenant-v2.0%20adopted-black.svg)](code_of_conduct.md)
[![Python Version](https://img.shields.io/badge/python-3.6%2C3.7%2C3.8-black.svg)](code_of_conduct.md)
![Pypi Version](https://img.shields.io/pypi/v/quickshow.svg)
![Code convention](https://img.shields.io/badge/code%20convention-pep8-black)

Quick-Show helps you draw plots quickly and easily. <br>
It is an abstraction using popular libraries such as Scikit-Learn and MatPlotLib, thus it is very light and convenient. <br><br>
`Note`: Quick-Show is sub-modules of other packages to manage quickshow more lightly and use more widly. 
*This is a project under development. With the end of the project, We plan to provide documents in major version 1 and sphinx. It is **NOT** recommended to use prior to major version 1.*

<br><Br>

# Installation
  ```cmd
  $ pip install quickshow
  ```
<br>

# Tutorial
1. Main-tutorials: https://github.com/DSDanielPark/quick-show/blob/main/tutorial/tutorial.ipynb
2. Sub-tutorial-folder: Tutorials for each function can be found in [this folder](https://github.com/DSDanielPark/quick-show/tree/main/tutorial). The tutorial is synchronized with the Python file name provided by QuickShow.
 

<Br>


# Features
## 1  Related to dimensionality reduction
2D or 3D t-SNE and PCA plots using specific columns of a refined dataframe. 
Create a scatter plot very quickly and easily by inputting a clean dataframe and column names that do not have missing data. 

1) `vis_tsne2d`: Simple visuallization of 2-dimensional t-distributed stochastic neighbor embedding <br>
2) `vis_tsne3d`: Simple visuallization of 3-dimensional t-distributed stochastic neighbor embedding <br>
3) `vis_pca`: Simple visuallization of Principal Component Analysis (PCA) 

<br>

## 2  Related to classification model evaluation. 
Later these functions are encapsulated into classes. <br>

3) `vis_cm`: Visuallization heatmap of confusion_matrix and return classification report dataframe. <br>
4) `get_total_cr_df`: When the confusion matrix dataframe created by the vis_cm function is saved as csv, the directory of the folder where these csv files exist is received as input and the confusion matrices of all csv files are merged into a single data frame. <br> 
5) `vis_multi_plot`: It takes the return dataframe of get_total_cr_df as input and draws a nice plot. However, if you want to use this function, please note that the suffix of the multiple csv files input to get_total_cr_df must be exp and an integer, such as `exp3`, and the integers must be `contiguous`.

<br>

## 3  Related to clustering. 

6) `vis_cluster_plot`: Produces a plot to see how spread out the actual label values ​​are within the clusters.<br>

<br>

## 4  Utils 

7) `find_all_files`: If you enter the top folder path as an auxiliary function, it returns a list of files including keywords while recursively searching subfolders. This is implemented with the glob package.<br>
8) `rcparam`: It simply shows some rcparams method in matploblib. Check by calling `qs.rcparam?`

<Br><Br>

# Examples
## Feature 1  <br>
  <details>
  <summary> See example dataframe... </summary>

  ```python
  import pandas as pd
  df = pd.DataFrame([3,2,3,2,3,3,1,1])
  df['val'] = [np.array([np.random.randint(0,10000),np.random.randint(0,10000),np.random.randint(0,10000)]) for x in df[0]]
  df.columns = ['labels', 'values']
  print(df)
  ```

  |    |   labels | values           |
  |---:|---------:|:-----------------|
  |  0 |        3 | [8231 3320 6894] |
  |  1 |        2 | [3485    7 7374] |
  |  ... |        ... |... |
  |  6 |        1 | [5218 9846 2488] |
  |  7 |        1 | [6661 5105  136] |

  </details>

  ```python
  from quickshow import vis_tsne2d, vis_tsne3d, vis_pca

  return_df = vis_tsne2d(df, 'values', 'labels', True, './save/fig1.png')
  return_df = vis_tsne3d(df, 'values', 'labels', True, './save/fig2.png')
  return_df = vis_pca(df, 'values', 'labels', 2, True, './save/fig3.png')
  return_df = vis_pca(df, 'values', 'labels', 3, True, './save/fig4.png')
  ```

  ![](https://github.com/DSDanielPark/quick-show/blob/main/quickshow/output/readme_fig1.png)
  ![](https://github.com/DSDanielPark/quick-show/blob/main/quickshow/output/readme_fig2.png)

  - All function returns the dataframe which used to plot. Thus, use the returned dataframe object to customize your plot. Or use [matplotlib's rcparam](https://matplotlib.org/stable/tutorials/introductory/customizing.html) methods.
  - If the label column does not exist, simply enter `None` as an argument.
  - For more details, please check doc string.
  
<br>

## Feature 2 
  <details>
  <summary> See example dataframe... </summary>

  ```python
  import pandas as pd
  label_list, num_rows = ['cat', 'dog', 'horse', 'dorphin'], 300
  df = pd.DataFrame([label_list[np.random.randint(4)] for _ in range(num_rows)], columns=['real'])
  df['predicted'] = [label_list[np.random.randint(4)] for _ in range(num_rows)]  
  print(df)
  ```

  |     | real    | predicted   |
  |----:|:--------|:------------|
  |   0 | cat     | cat         |
  |   1 | horse   | cat         |
  | ... | ...     | ...         |
  |   7 | horse   | dog         |
  | 299 | dorphin | horse       |

  </details>

  ```python
  from quickshow import vis_cm

  df_cr, cm = vis_cm(df, 'real', 'predicted', 'vis_cm.csv', 'vis_cm.png')
  ```


  ```python
  print(df_cr)
  ```
  |           |       cat |       dog |   dorphin |     horse |   accuracy |   macro avg |   weighted avg |
  |:----------|----------:|----------:|----------:|----------:|-----------:|------------:|---------------:|
  | precision |  0.304878 |  0.344828 |  0.285714 |  0.276316 |        0.3 |    0.302934 |       0.304337 |
  | recall    |  0.328947 |  0.246914 |  0.328767 |  0.3      |        0.3 |    0.301157 |       0.3      |
  | f1-score  |  0.316456 |  0.28777  |  0.305732 |  0.287671 |        0.3 |    0.299407 |       0.299385 |
  | support   | 76        | 81        | 73        | 70        |        0.3 |  300        |     300        |


  confusion matirx will be shown as below.
  ![](https://github.com/DSDanielPark/quick-show/blob/main/quickshow/output/readme_fig3.png)

  - This function return pandas.DataFrame obejct of classification report and confusion metix as shown below.
  

<br>
<br>

# Use Case
[1] [Korean-news-topic-classification-using-KO-BERT](https://github.com/DSDanielPark/fine-tuned-korean-BERT-news-article-classifier): all plots were created through Quick-Show.

# References
[1] Scikit-Learn https://scikit-learn.org <br>
[2] Matplotlib https://matplotlib.org/


### Contacts
Maintainers: [Daniel Park, South Korea](https://github.com/DSDanielPark) 
e-mail parkminwoo1991@gmail.com
