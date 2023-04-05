from clip_retrieval.clip_back import load_clip_index, load_clip_indices, KnnService, ClipOptions 
import pandas as pd
import argparse

META_DATA_COLS = ["url", "caption", "image_path"]

def get_knn_service(index_path=str, name=str, model="ViT-L/14"):
    """
    get knn instance based on given index.

    @param indices_path: path of index
    @return: knn instance
    """
    clip_dict = {}
    single_index = load_clip_index(
        indices_paths=index_path,
        clip_options=ClipOptions(
            indice_folder="",
            clip_model=model,
            enable_hdf5=False,
            enable_faiss_memory_mapping=False,
            columns_to_return=META_DATA_COLS,
            reorder_metadata_by_ivf_index=False,
            enable_mclip_option=False,
            use_jit=True,
            use_arrow=False,
            provide_aesthetic_embeddings=False,
            provide_safety_model=False,
            provide_violence_detector=False,
        )
    )
    
    clip_dict[name] = single_index
    return KnnService(
        clip_resources=clip_dict
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("index_path", type=str,
                        help="path of index to be loaded.")
    parser.add_argument("query_folder", type=str,
                        help="folder path of query images.")
    parser.add_argument("num", type=int,
                        help="number of output results.")
    parser.add_argument("--model", type=str,
                        help="model used in index, ViT-B/32 or ViT-L/14")
    
    args = parser.parse_args()
    #
    if args.model:
        knn_service = get_knn_service(args.index_path, "laion400m", args.model)
    else:
        knn_service = get_knn_service(args.index_path, "laion400m")
    
    results = knn_service.multi_img_query(
        image_folder=args.query_folder,
        model=args.model if args.model else "ViT-L/14",
        modality="image",
        num_images=args.num,
        num_result_ids=args.num,
        indice_name=None, # meaningful only when multiple indices were loaded
        deduplicate=True, # whether to eliminate duplicated results
    )

    #
    assert results != None, f"For given images no query results can be retrieved. Pls check the given parameters."

    for idx, result in results:
        res_table = pd.DataFrame(
            [(e['image_path'], e['id'], e['similarity']) for e in result],
            columns=["image_path", "id", "similarity"],
        )
        print("====================================")
        print("======= results for image {} =======".format(idx))
        print(res_table)

    