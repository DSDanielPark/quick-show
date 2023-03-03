import seaborn as sns
import matplotlib.pyplot as plt


sns.set()  # use Seaborn styles
plt.rcParams["figure.figsize"] = (16,10)


def vis_cluster_plot(df, cluster_col, label_col, save_path):
    bin_list = [0]
    total = 0
    for i in range(16):
        count = len(df[df[[cluster_col,label_col]][cluster_col] == i][label_col].unique())
        total += count
        bin_list.append(total-1)
    df_pivot =df.pivot_table(index=[cluster_col, label_col], columns=label_col, aggfunc='size', fill_value=0)
    df_pivot.plot()

    plt.ylabel('Count of ground truth category', fontsize=20)
    plt.xlabel('Cluster bin', fontsize=20)
    plt.tick_params(
        axis='x',     
        which='both',    
        bottom=False,     
        top=False,         
        labelbottom=False)
    plt.title("how many real label is ditributed in each cluster", fontsize=20)
    plt.grid(True, axis='x', color='b',linewidth=1)
    plt.xticks(bin_list, label=False)
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return df_pivot
    