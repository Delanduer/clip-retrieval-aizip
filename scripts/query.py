from clip_retrieval.clip_back import load_clip_index, load_clip_indices, KnnService, ClipOptions 
import pandas as pd
import argparse
import time

META_DATA_COLS = ["url", "caption", "image_path"]

def get_knn_service(args=argparse.Namespace, name=str):
    """
    get knn instance based on given index.

    @param indices_path: path of index
    @return: knn instance
    """
    clip_dict = {}
    single_index = load_clip_index(
        clip_options=ClipOptions(
            indice_folder=args.index_path,
            clip_model=args.model,
            enable_hdf5=False,
            enable_faiss_memory_mapping=False,
            columns_to_return=META_DATA_COLS,
            reorder_metadata_by_ivf_index=False,
            enable_mclip_option=False,
            use_jit=True,
            use_arrow=False,
            provide_aesthetic_embeddings=args.aesthetic_embeddings if args.aesthetic_embeddings else False,
            provide_safety_model=args.safety_model if args.safety_model else False,
            provide_violence_detector=args.violence_detector if args.violence_detector else False,
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
    parser.add_argument("--aesthetic_embeddings", type=bool,
                        help="whether to add aesthetics")
    parser.add_argument("--safety_model", type=bool,
                        help="whether to filter out unsafe items")
    parser.add_argument("--violence_detector", type=bool,
                        help="whether to filter out items with violence content")
    
    args = parser.parse_args()
    
    knn_load_start = time.perf_counter()
    knn_service = get_knn_service(args=args, name="laion400m")
    knn_load_end = time.perf_counter()
    print("Index loading duration: {}".format(knn_load_end-knn_load_start))
    results = knn_service.multi_img_query(
        image_folder=args.query_folder,
        model=args.model if args.model else "ViT-L/14",
        modality="image",
        num_images=args.num,
        num_result_ids=args.num,
        indice_name=None, # meaningful only when multiple indices were loaded
        deduplicate=True, # whether to eliminate duplicated results
        use_safety_model=False, # requires "safety_model" set to True
        use_violence_detector=False, # requires "violence_detector" set to True
        aesthetic_score=None, # requires "aesthetic_embeddings" set to True
        aesthetic_weight=None, # requires "aesthetic_embeddings" set to True
    )
    knn_query_end = time.perf_counter()
    print("Actual query duration: {}".format(knn_query_end-knn_load_end))

    assert results != None, f"For given images no query results can be retrieved. Pls check the given parameters."

    for idx, result in enumerate(results):
        res_table = pd.DataFrame(
            [(e['image_path'], e['id'], e['similarity']) for e in result],
            columns=["image_path", "id", "similarity"],
        )
        print("====================================")
        print("======= results for image {} =======".format(idx))
        print(res_table)

    