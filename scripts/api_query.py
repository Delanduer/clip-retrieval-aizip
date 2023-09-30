from aizip_clip_query import ClipClassifierQuery, ClipImgQuery
import pandas as pd

classifier_query = ClipClassifierQuery(
    host_uri = "http://127.0.0.1",
    port = "13005",
)

img_query = ClipImgQuery(
    host_uri = "http://127.0.0.1",
    port = "13005",
)

image_q = img_query.query_for_single_image(
    image_url = "/home/junjie/junjie/clip_retrieval/testimages/indian-bride-1181855.jpg",
    print_res = True,
)

image_batch = img_query.query_for_image_dir(
    dir = "/home/junjie/junjie/9497_images",
    print_res = True,
)