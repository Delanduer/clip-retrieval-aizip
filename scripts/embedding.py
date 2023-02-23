import os
from clip_retrieval.clip_inference.main import main

def getsubfolderlist(parent_folder):
    print("Parent folder: {}".format(parent_folder))
    from glob import glob
    subfolders = glob(parent_folder + "/*/", recursive=True)
    print("Total number of subfolders detected: {}\nFirst subfolder: {}".format(len(subfolders),subfolders[0]))
    return subfolders


def callinference():
    main(
        #input_dataset=["/home/junjie/git/clip-retrieval/notebook/cat_test/multiple/", "/home/junjie/git/clip-retrieval/notebook/cat_test/multiple_1/"],
        input_dataset=getsubfolderlist("/nfs/ssd4/data/laion400m/i24"),
        #input_dataset=["/ssd/mlrom/Data/coco2017/val2017", "/ssd/mlrom/Data/coco2017/train2017", "/ssd/mlrom/Data/vww/all2014"],
        #input_dataset=["/ssd/mlrom/Data/vww/all2014"],
        output_folder="/home/junjie/test/multitaskgpupara/emb_laioni24_l14_b1024_700",
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
    #_ = getsubfolderlist("/nfs/ssd4/data/laion400m/i24")   # for testing
    callinference()
