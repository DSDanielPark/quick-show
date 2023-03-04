import matplotlib.pyplot as plt
import os


def find_all_files(folder_path:str, include_word:str) -> list:
    all_file_list = []
    for (root, directories, files) in os.walk(folder_path):
        for file in files:
            if include_word in file:
                file_path = os.path.join(root, file)
                all_file_list.append(file_path)

    return all_file_list


def rcparam(**args):
    plt.style.use('tableau-colorblind10')
    plt.rcParams["figure.figsize"] = (20,10)
    plt.rcParams['lines.linewidth'] = 4
    plt.rcParams['axes.grid'] = False 
    plt.rcParams['axes.facecolor'] = 'white'

    return None

        
