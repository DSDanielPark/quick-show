# Quick-Show
- Quick-Show is a package that allows you to easily and quickly draw 2D or 3D t-SNE and PCA plots using specific columns of a refined dataframe.
- Quick Show is an abstraction using popular libraries such as sklearn and matplotlib, so it is very light and convenient.
- `Note`: quich show is sub-modules of other packages to manage quickshow more lightly and use more widly.
- 추가 업데이트 계획이 있으므로, 간단한 함수로 관리하며, 추가 배포 예정 레포의 서브 모듈로 사용함.
<br>

## Quick Start
  ```cmd
  $ pip install quickshow
  ```
- Create a scatter plot very quickly and easily by inputting a clean dataframe and column names that do not have missing data. 
- If the label column does not exist, simply enter `None` as an argument.
  ```python
  from quickshow import vis_tsne2d, vis_tsne3d, vis_pca

  # Make sample df
  df = pd.DataFrame([3,2,3,2,3,3,1,1])
  df['val'] = [np.array([np.random.randint(0,10000),np.random.randint(0,10000),np.random.randint(0,10000)]) for x in df[0]]
  df.columns = ['labels', 'values']

  # Use matplotlib rcparams or returned dataframe for customize your plot.
  return_df = vis_tsne2d(df, 'values', 'labels', False, 'fig1.png')
  return_df = vis_tsne3d(df, 'values', 'labels', False, 'fig2.png')
  return_df = vis_pca(df, 'values', 'labels', 2, False, 'fig3.png')
  return_df = vis_pca(df, 'values', 'labels', 3, False, 'fig4.png')
  ```
  
  ![](https://github.com/DSDanielPark/quick-show/blob/main/quickshow/output/readme_fig1.png)
  ![](https://github.com/DSDanielPark/quick-show/blob/main/quickshow/output/readme_fig2.png)
  
  <!-- <img src="https://github.com/DSDanielPark/quick-show/blob/main/quickshow/output/readme_fig1.png" width="500"><BR>
  <img src="https://github.com/DSDanielPark/quick-show/blob/main/quickshow/output/readme_fig2.png" width="500"><BR> -->
  - For more details, please check doc string.
<br>
<br>


## Functions
It contains 3 functions: `vis_tsne2d`, `vis_tsne3d`, `vis_pca`
- (1) `vis_tsne2d` function: Simple visuallization of 2-dimensional t-distributed stochastic neighbor embedding
- (2) `vis_tsne3d` function: Simple visuallization of 3-dimensional t-distributed stochastic neighbor embedding
- (3) `vis_pca` function: Simple visuallization of Principal Component Analysis (PCA)


All function returns the dataframe which used to plot. Thus, use the returned dataframe object to customize your plot. Or use [matplotlib's rcparam](https://matplotlib.org/stable/tutorials/introductory/customizing.html) methods.
<br>
<br>


## References
[1] sklearn.manifold.TSNE https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html <br>
[2] sklearn.decomposition.PCA https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html <br>
[3] matplotlib https://matplotlib.org/
