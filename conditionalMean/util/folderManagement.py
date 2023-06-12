import os


def makeRecursiveFolder(path):
    folder_list = path.split("/")
    localFolder = ""
    for folder in folder_list:
        localFolder = os.path.join(localFolder, folder)
        os.makedirs(localFolder, exist_ok=True)
