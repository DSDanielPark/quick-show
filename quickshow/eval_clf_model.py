"""Create classification report dataframe and show confusion matrix.

Abbreviation
=========== ========================================================
Shorthand    full name
=========== ========================================================
cr           classification report
cm           confusion matrix
=========== ========================================================
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from quickshow.utils import find_all_files
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def vis_cm(df: pd.DataFrame, true_label_col: str, predicted_col: str, save_cr_csv_path: str, save_cmplot_path: str):
    """Create a heatmap using predicted column, ground truth label column.
    cr == classification report
    cm == confusion matrix

    Example code :
    df_cr, cm = vis_cm(df, 'real', 'predicted', './csv/exp1_cm.csv, './result/cm.png')
    vis_cm(df, 'ground_truth', 'predicted_y', None, None)

    Parameters
    ----------
    df : pd.DataFrame 
        pd.Dataframe object includes true label column and predicted_label column. 
    true_label_col : str
        column name which has ground_truth_labels.
    predicted_col : str
        column name which has predicted labels.
    save_cr_csv_path: str
        if you want to save df_cr, enter the full path. ('./result/df_cr.csv', or enter None)
    save_cmplot_path: str
        if you want to save cmplot image, enter the full path. ('./fig/cm.png' or enter None)
    """
    y_true = df[true_label_col]
    y_pred = df[predicted_col]
    classes = y_true.unique()

    try:
        cr = classification_report(y_true, y_pred, output_dict=True) # create classification report
        df_cr = pd.DataFrame(cr)
    except Exception as e:
        print(e)
    try:
        cm = confusion_matrix(df[true_label_col], df[predicted_col])
    except Exception as e: 
        print(e)

    plt.title("Confusion Metirx: True Label vs Predicted Label", fontsize =13, pad=20)
    sns.set_style("dark")
    ax = sns.heatmap(cm, cmap="crest", fmt=".3g", annot=True, xticklabels=classes, yticklabels=classes)
    ax.set(xlabel='common xlabel', ylabel='common ylabel')
    plt.xlabel("Predicted", labelpad=10)
    plt.ylabel("True Label", labelpad=10)
    plt.show()
    if save_cmplot_path is not None:
        plt.savefig(save_cmplot_path, dpi=300, bbox_inches = "tight")
    if save_cr_csv_path is not None:
        df_cr.to_csv(save_cr_csv_path, encoding='utf-8-sig', index=False) 
    plt.clf()
    
    return df_cr, cm


def get_total_cr_df(cm_csv_folder_path: str, include_word: str, save_path: str) -> list:
    metric_df_list = find_all_files(cm_csv_folder_path, include_word)
    for i in range(len(metric_df_list)):
        if i == 0:
            total_cr_df = pd.read_csv(metric_df_list[i])
            exp_name = metric_df_list[i].split('\\')[-1].split('_')[0]
            total_cr_df['exp'] = exp_name
        else:
            exp_name = metric_df_list[i].split('\\')[-1].split('_')[0]
            df = pd.read_csv(metric_df_list[i])
            df['exp'] = exp_name
            total_cr_df = pd.concat([df, total_cr_df], ignore_index=True, axis=0)
    total_cr_df['exp'] = [int(x.lstrip('exp')) for x in total_cr_df['exp']]

    if save_path is not None:
        total_cr_df.to_csv(save_path, index=False, encoding='utf-8-sig')
    total_cr_df.rename(columns = {'Unnamed: 0' : 'metric'}, inplace = True)

    return total_cr_df


def vis_multi_plot(df: pd.DataFrame, which_metirc: str, except_col: str, save_path: str) -> pd.DataFrame:
    df = df[df['metric'] == which_metirc]
    df = df.sort_values(by='exp' ,ascending=False)
    for i, col in enumerate(df.columns):
        if col not in except_col:
            plt.plot(df.exp, df[col], '-o', label=col, linewidth=1)
            plt.legend(loc="lower right")
        plt.grid(color='gray', linewidth=1)
        plt.legend
    plt.title(f"X: number of experiments, Y: {which_metirc}", fontsize=25, pad=40)
    plt.xlabel('Number of Experiments',  fontsize=20, labelpad=30)
    plt.ylabel(f'{which_metirc} for each class', fontsize=20, labelpad=30)
    plt.show()
    if save_path is not None:
        plt.savefig(save_path ,dpi=300, bbox_inches = "tight")

    return df