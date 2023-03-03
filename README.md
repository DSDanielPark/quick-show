
# Quick-Show

[![Contributor Covenant](https://img.shields.io/badge/contributor%20covenant-v2.0%20adopted-green.svg)](code_of_conduct.md)
[![Python Version](https://img.shields.io/badge/python-3.6%2C3.7%2C3.8-blue.svg)](code_of_conduct.md)
![Pypi Version](https://img.shields.io/pypi/v/quickshow.svg)
![Code convention](https://img.shields.io/badge/code%20convention-pep8-violet)

Quick-Show is a package that allows you to easily and quickly draw plots. <br>
Quick Show is an abstraction using popular libraries such as sklearn and matplotlib, so it is very light and convenient. <br><br>
`Note`: Quick-Show is sub-modules of other packages to manage quickshow more lightly and use more widly. 
*This is a project under development as a submodule. With the end of the project, We plan to provide documents in major version 1 and sphinx. It is **NOT** recommended to use prior to major version 1.*

<br>

# Installation
  ```cmd
  $ pip install quickshow
  ```
<br>
 
# Guide
## 1  Plots related to dimensionality reduction
2D or 3D t-SNE and PCA plots using specific columns of a refined dataframe. 
Create a scatter plot very quickly and easily by inputting a clean dataframe and column names that do not have missing data. 
<br>

### 1.1 Functions <br>

1) `vis_tsne2d` function: Simple visuallization of 2-dimensional t-distributed stochastic neighbor embedding <br>
2) `vis_tsne3d` function: Simple visuallization of 3-dimensional t-distributed stochastic neighbor embedding <br>
3) `vis_pca` function: Simple visuallization of Principal Component Analysis (PCA) <br><br>

### 1.2 Example <br>
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

  <details>
  <summary> See output figure... </summary>

  ![](https://github.com/DSDanielPark/quick-show/blob/main/quickshow/output/readme_fig1.png)
  ![](https://github.com/DSDanielPark/quick-show/blob/main/quickshow/output/readme_fig2.png)

  - All function returns the dataframe which used to plot. Thus, use the returned dataframe object to customize your plot. Or use [matplotlib's rcparam](https://matplotlib.org/stable/tutorials/introductory/customizing.html) methods.
  - If the label column does not exist, simply enter `None` as an argument.
  - For more details, please check doc string.
  
  </details>
<br>






## 2  Plots related to classification model evaluation. 

<br>

### 2.1 Functions <br>

1) `vis_cm` function: visuallization heatmap of confusion_matrix and return classification report dataframe. <br>

### 2.2 Examples <br><br>

  `vis_cm`
  
  <details>
  <summary> See example dataframe... </summary>

  ```python
  import pandas as pd
  label_list, num_rows = ['cat', 'dog', 'hores', 'dorphin'], 300
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

  <details>
  <summary> See output... </summary>

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
  
  </details>
<br>
<br>
<br>



# References
[1] Scikit-Learn https://scikit-learn.org <br>
[2] Matplotlib https://matplotlib.org/
<br>

<br>

### Contacts
Project Owner(P.O): [Daniel Park, South Korea](https://github.com/DSDanielPark) 
e-mail parkminwoo1991@gmail.com <br>
Maintainers: [Daniel Park, South Korea](https://github.com/DSDanielPark) 
e-mail parkminwoo1991@gmail.com
