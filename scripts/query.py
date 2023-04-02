from clip_retrieval.clip_back import load_clip_index, load_clip_indices, KnnService, ClipOptions 
import pandas as pd

META_DATA_COLS = ["url", "caption", "image_path"]

def get_knn_service(indices_path):
    """
    """
    clip_dict = {}
    single_index = load_clip_index(
        indices_paths=indices_path,
        clip_options=ClipOptions(
            indice_folder="",
            clip_model="ViT-L/14",
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
    
    clip_dict["laion400m"] = single_index
    return KnnService(
        clip_resources=clip_dict
    )

if __name__ == "__main__":
    #
    index_path = ""
    query_img_folder = ""
    num_results = 5

    #
    knn_service = get_knn_service(index_path)
    
    results = knn_service.multi_img_query(
        image_folder=query_img_folder,
        model="ViT-L/14",
        modality="image",
        num_images=num_results,
        num_result_ids=num_results,
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

    