from qs import *
import pandas as pd
import numpy as np


if __name__ == "__main__":
    df = pd.DataFrame([3,2,3,2,3,3,1,1])
    df['val'] = [np.array([np.random.randint(0,10000),np.random.randint(0,10000),np.random.randint(0,10000)]) for x in df[0]]
    df.columns = ['labels', 'values']
    return_df = vis_tsne2d(df, 'values', 'labels', False, 'fig1.png')
    return_df = vis_tsne3d(df, 'values', 'labels', False, 'fig2.png')
    return_df = vis_pca(df, 'values', 'labels', 2, False, 'fig3.png')
    return_df = vis_pca(df, 'values', 'labels', 3, False, 'fig4.png')
