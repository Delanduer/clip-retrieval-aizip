import sys

#sys.path.insert(
#    0,
#    r'C:\Users\wjjbf\Documents\Git\clip-retrieval-aizip\aizip_clip_server_packaging\aizip_clip_server',
#)

#from aizip_clip_back import clip_back_fastapi
from aizip_clip_server import clip_back_fastapi

columns = ["url", "caption", "image_path"]
app = clip_back_fastapi(
    index_folder=r"C:\\Users\\wjjbf\\Documents\\Git\\clip-retrieval-aizip\\test_index_local\\laion400m_index1",
    columns_to_return=columns,
    reorder_metadata_by_ivf_index=False,
    enable_mclip_option=True,
    clip_model="ViT-L/14",
    use_jit=True,
    use_arrow=False,
    provide_safety_model=False,
    provide_violence_detector=False,
    provide_aesthetic_embeddings=True,
)

import uvicorn
uvicorn.run(
    app=app,
    #"clip_backend:app",
    host="127.0.0.1",
    port=13005,
    #reload=True,
)