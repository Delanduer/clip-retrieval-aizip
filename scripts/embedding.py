import os
from clip_retrieval.clip_inference.main import main

def getfirstlvlsubfolderlist(parent_folder):
    from glob import glob
    subfolders = glob(parent_folder + "/*/", recursive=True)
    print("Total number of subfolders detected: {} in parent folder: {}\nFirst subfolder: {}".format(len(subfolders),parent_folder, subfolders[0]))
    return subfolders

def getfoldersfrompattern(path_list, pattern):
    if len(path_list) < 1:
        print("Error: given path list is empty for extract folder with pattern.")
    else:
        output_list = []
        for path in path_list:
            subfolders = getfirstlvlsubfolderlist(path)
            for subfolder in subfolders:
                if pattern in subfolder:
                    output_list.append(subfolder)
        return output_list

def getimgfolders(path_list):
    if len(path_list) < 1:
        print("Error: given path list is empty for getting img folders.")
        return []
    else:
        output_list = []
        for path in path_list:
            loowest_name = path.rsplit("/", 2)[1]
            if lowest_name == []:
                print("Error when extracting lowest path name for: {}".format(path))
                continue
            else:
                img_idx = lowest_name.rsplit("-", 2)[1]
                new_path = os.path.join(path, "i"+img_idx)
                output_list.extend(getfirstlvlsubfolderlist(new_path))
        return output_list

def callinference(input_datasets, output_folder):
    main(
        input_dataset=input_datasets,
        output_folder=output_folder,
        batch_size=1024,  # 768 causes CUDA memoray error
        #clip_model="ViT-B/32",   #ViT-B/32, ViT-L/14
        clip_model="ViT-L/14",
        use_jit=True,
        distribution_strategy="sequential",
        enable_text=False,
        enable_image=True,
        enable_metadata=False
    )


if __name__  == "__main__":
    #ssdlist=[]
    #for i in range(2, 10):
    #     ssd_path = "/nfs/ssd" + str(i) + "/data/laion400m"
    #     ssdlist.append(ssd_path)
    #print("Total number of ssd paths: {}".format(len(ssdlist)))
    #parent_datasets = getfoldersfrompattern(ssdlist, "-img")
    #datasets = getimgfolders(parent_datasets)
    
    task = "29"
    print("Task for: {}".format(task))
    imgfolder = "/nfs/ssd9/data/laion400m/laion400m-data-" + task + "-img/i" + task
    datasets = getfirstlvlsubfolderlist(imgfolder)
    output_folder = "/nfs/ssd14/projects/junjie/laion400mi" + task

    callinference(datasets, output_folder)
