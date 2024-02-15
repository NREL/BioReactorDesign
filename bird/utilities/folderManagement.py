import os
import re


def makeRecursiveFolder(path):
    folder_list = path.split("/")
    localFolder = ""
    for folder in folder_list:
        localFolder = os.path.join(localFolder, folder)
        os.makedirs(localFolder, exist_ok=True)


def getManyFolders(rootFolder, prefix="flat_donut"):
    # Read Time
    fold_tmp = os.listdir(rootFolder)
    fold_num = []
    # remove non floats
    for i, entry in reversed(list(enumerate(fold_tmp))):
        if not entry.startswith(prefix) or entry.endswith("N2"):
            a = fold_tmp.pop(i)
            # print('removed ', a)
    for entry in fold_tmp:
        num = re.findall(r"\d+", entry)
        if len(num) > 1:
            print(f"WARNING: Cannot find num of folder {entry}.")
            print("Do not trust the spearman stat")
        else:
            fold_num.append(int(num[0]))

    sortedFold = [
        x for _, x in sorted(zip(fold_num, fold_tmp), key=lambda pair: pair[0])
    ]
    return sortedFold
