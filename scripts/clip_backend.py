#import sys
#sys.path.insert(0, '../aizip_clip_server_packaging/aizip_clip_query')
from aizip_clip_server import clip_back_fastapi

columns = ["url", "caption", "image_path"]
#index_parent_path = "/nfs/ssd14/projects/junjie/laion400m_index_tests/10ms" # PQ128 51G
index_parent_path = "/home/junjie/junjie/index_factory/laion_PQ48"
#index_parent_path = "/nfs/ssd14/projects/junjie/laion400m_index1"

app = clip_back_fastapi(
    index_folder=index_parent_path,
    columns_to_return=columns,
    reorder_metadata_by_ivf_index=False,
    enable_mclip_option=True,
    clip_model="ViT-L/14",
    use_jit=True,
    use_arrow=False,
    provide_safety_model=False,
    provide_violence_detector=False,
    provide_aesthetic_embeddings=True,
    load_in_gpu=True,
    ngpu=1,
)

import uvicorn
uvicorn.run(
    app=app,
    #"clip_backend:app",
    host="127.0.0.1",
    port=13005,
    #reload=True,
)