import os
from clip_retrieval.clip_index import clip_index

def callindex():
    clip_index(
        embeddings_folder="/nfs/ssd14/projects/junjie/laion400m",
        index_folder="/nfs/ssd14/projects/junjie/laion400m_index",
        copy_metadata=True,
        max_index_memory_usage="16G",
        current_memory_available="64G",
    )



if __name__=="__main__":
    callindex()
