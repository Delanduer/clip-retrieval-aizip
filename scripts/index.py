import os
from clip_retrieval.clip_index import clip_index

def callindex():
    clip_index(
        embeddings_folder="/home/junjie/test/multitaskgpupara/emb_coco_1024b_l14_writeb1000_trainvalvww2014",
        index_folder="/home/junjie/test/multitaskgpupara/inx_coco_1024b_l14_writeb1000_trainvalvww2014",
        copy_metadata=True
    )



if __name__=="__main__":
    callindex()
