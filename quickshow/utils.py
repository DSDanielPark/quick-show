import os


def find_all_files(folder_path:str, include_word:str) -> list:
    all_file_list = []
    for (root, directories, files) in os.walk(folder_path):
        for file in files:
            if include_word in file:
                file_path = os.path.join(root, file)
                all_file_list.append(file_path)

    return all_file_list