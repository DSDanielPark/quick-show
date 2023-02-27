"""
A simple visualization class.
Create and visualize dimensionally reduced pca and tsne columns through a specific column of clean dataframe.

=========== ========================================================
Shorthand    full name
=========== ========================================================
t-SNE        t-distributed Stochastic Neighbor Embedding
PCA          Principal Component Analysis
=========== ========================================================
"""


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams["figure.figsize"] = (7,7)
plt.rcParams['lines.linewidth'] = 3
# plt.rcParams['axes.grid'] = True 
# plt.rcParams['axes.facecolor'] = 'gray'


def vis_tsne2d(df: pd.DataFrame, target_col: str, label_col: str, show_plot: bool, save_plot_path: str) -> pd.DataFrame:
    """Create a 2d tsne columns through t-SNE(t-distributed Stochastic Neighbor Embedding) and simply visualize it.

    Example code :
    
    return_df = vis_tsne2d(embedded_df, 'target_col_embedding', 'category', True, './result/fig1.png')
    return_df = vis_tsne2d(input_df, 'token_array', 'bin_number', False, None)

    Parameters
    ----------
    df : pd.DataFrame 
        Data frame with target_col and label_col. 
    target_col : str
        Data column name to process pca included in data frame.
    label_col : str
        Label column name to be used for visualization if target_col has different labels.
        Enter None if label is missing or identical. (`None` or `string of column name`)
    save_plot_path: str
        Enter the full path to save result image.
    """
    data_subset = df[target_col].values
    raveled_data_subset = [x.ravel() for x in data_subset]

    tsne = TSNE(n_components=2, verbose=10, perplexity=3, n_iter=500)
    df_tsne_2d = tsne.fit_transform(raveled_data_subset)
    df['tsne-2d-1'] = df_tsne_2d[:,0]
    df['tsne-2d-2'] = df_tsne_2d[:,1]
    
    sns.scatterplot(
        x="tsne-2d-1", y="tsne-2d-2",
        hue=label_col,
        palette=sns.color_palette("Set2", len(df[label_col].unique())),
        data=df,
        legend="full",
        alpha=0.3
    )
    if save_plot_path is not None:
        plt.show()
        plt.savefig(save_plot_path, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.clf()
    
    return df


def vis_tsne3d(df: pd.DataFrame, target_col: str, label_col: str, show_plot: bool, save_plot_path: str) -> pd.DataFrame:
    """Create a 3d tsne columns through t-SNE(t-distributed Stochastic Neighbor Embedding) and simply visualize it.

    Example code :
    qs = QuickShow()
    return_df = qs.vis_tsne3d(embedded_df, 'target_col_embedding', 'category', True, './result/fig1.png')
    return_df = qs.vis_tsne3d(input_df, 'token_array', 'bin_number', False, None)
    
    Parameters
    ----------
    df : pd.DataFrame 
        Data frame with target_col and label_col. 
    target_col : str
        Data column name to process pca included in data frame.
    label_col : str
        Label column name to be used for visualization if target_col has different labels.
        Enter None if label is missing or identical. (`None` or `string of column name`)
    save_plot_path: str
        Enter the full path to save result image.
    """
    data_subset = df[target_col].values
    raveled_data_subset = [x.ravel() for x in data_subset]

    tsne = TSNE(n_components=3, verbose=10, perplexity=3, n_iter=300)
    df_tsne_3d = tsne.fit_transform(raveled_data_subset)
    df['tsne-3d-1'] = df_tsne_3d[:,0]
    df['tsne-3d-2'] = df_tsne_3d[:,1]
    df['tsne-3d-3'] = df_tsne_3d[:,2]

    x, y, z = df['tsne-3d-1'], df['tsne-3d-2'], df['tsne-3d-3']
    ax = plt.axes(projection ="3d")
    ax.scatter3D(x, y, z)
    plt.title("T-SNE 3D")
    for s in df[label_col].unique():
        ax.scatter(df['tsne-3d-1'][df[label_col]==s], 
                df['tsne-3d-2'][df[label_col]==s], 
                df['tsne-3d-3'][df[label_col]==s], 
                label=s)
    ax.set_xlabel('tsne-3d-1')
    ax.set_ylabel('tsne-3d-2')
    ax.set_zlabel('tsne-3d-3')
    # ax.set(ylim=(-0, 30))
    # ax.set(xlim=(-150, 30))
    # ax.set(zlim=(-200, 50))
    ax.legend()

    if save_plot_path is not None:
        plt.show()
        plt.savefig(save_plot_path, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.clf()

    return df
        

def vis_pca(df: pd.DataFrame, target_col: str, label_col: str, pca_dim: int, show_plot: bool, save_plot_path: str) -> pd.DataFrame:
    """Create a pc columns through sklearn.decomposition and simply visualize it.

    Example code :
    qs = QuickShow()
    return_df = qs.vis_pca(embedded_df, 'target_col_embedding', 'category', 2, True, './result/fig1.png')
    return_df = qs.vis_pca(input_df, 'token_array', 'bin_number', 3, False, None)

    Parameters
    ----------
    df : pd.DataFrame 
        Data frame with target_col and label_col. 
    target_col : str
        Data column name to process pca included in data frame.
    label_col : str
        Label column name to be used for visualization if target_col has different labels.
        Enter None if label is missing or identical. (`None` or `string of column name`)
    pca_dim : int
        Choose a dimension to visualize. (Enter 2 or 3 only)
    save_plot_path: str
        Enter the full path to save result image.
    """
    data_subset = df[target_col].values
    raveled_data_subset = [x.ravel() for x in data_subset]

    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(raveled_data_subset)
    df['PC1'] = pca_result[:,0]
    df['PC2'] = pca_result[:,1] 
    df['PC3'] = pca_result[:,2]
    print('Explained variation per PC(Principal Component): {}'.format(pca.explained_variance_ratio_))
    
    if pca_dim == 2:
        sns.scatterplot(
            x="PC1", y="PC2",
            hue=label_col,
            palette=sns.color_palette("Set2", len(df[label_col].unique())),
            data=df,
            legend="full",
            alpha=0.3
        )
        if save_plot_path is not None:
            plt.savefig(save_plot_path, bbox_inches='tight')
    elif pca_dim == 3:
        ax = plt.figure(figsize=(10,10)).gca(projection='3d')
        if label_col is None:
            ax.scatter(df['PC1'], df['pa-2'], df['PC3']) 
        else:
            for s in df[label_col].unique():
                ax.scatter(df['PC1'][df[label_col]==s], 
                        df['PC2'][df[label_col]==s], 
                        df['PC3'][df[label_col]==s], 
                        label=s)
        ax.legend()
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        if save_plot_path is not None:
            plt.show()
            plt.savefig(save_plot_path, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.clf()

    return df



