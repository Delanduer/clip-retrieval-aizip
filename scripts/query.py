from clip_retrieval.clip_back import load_clip_indices, KnnService, ClipOptions 
import pandas as pd

META_DATA_COLS = ["url", "caption", "image_path"]

def load_indices(indices_path):
    return load_clip_indices(
        indices_paths=indices_path,
        clip_options=ClipOptions(
            indice_folder="",
            clip_model="ViT-L/64",
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


if __name__ == "__main__":
    #
    indices_path = ""
    query_img_folder = ""
    num_results = 5

    #
    indices_resource = load_indices(indices_path)
    print("Loaded clip recource has following keys:\n{}".format(indices_resource.keys()))

    #
    knn_service = KnnService(
        clip_resources=indices_resource
    )
    results = knn_service.multi_img_query(
        image_folder=query_img_folder,
        model="ViT-L/64",
        modality="image",
        num_images=num_results,
        num_result_ids=num_results,
        indice_name=None, # meaningful only when multiple indices were loaded
        deduplicate=True
    )

    #
    res_table = pd.DataFrame(
        [(e['image_path'], e['id'], e['similarity']) for e in results],
        columns=["image_path", "id", "similarity"],
    )
    print("=======================")
    print("======= results =======")
    print(res_table)

    