from clustering import *
def folder(folder_name):
    for file in os.listdir(folder_name):
        if "thresh" in file:
            slices(folder_name, file)
    cluster("./Slices/")
folder("testPatient")