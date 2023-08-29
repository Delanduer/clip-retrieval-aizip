"""Clip back: host a knn service using clip as an encoder"""


from typing import Callable, Dict, Any, List
from flask import request, make_response
from flask_restful import Resource, Api
import faiss
from collections import defaultdict
from multiprocessing.pool import ThreadPool
import json
from io import BytesIO
from PIL import Image
import base64
import os
import fire
from pathlib import Path
import pandas as pd
import urllib
import tempfile
import io
import numpy as np
from functools import lru_cache
import pyarrow as pa
import fsspec

import h5py
from tqdm import tqdm
from prometheus_client import Histogram, REGISTRY, make_wsgi_app
import math
import logging

from clip_retrieval.ivf_metadata_ordering import (
    Hdf5Sink,
    external_sort_parquet,
    get_old_to_new_mapping,
    re_order_parquet,
)
from dataclasses import dataclass

from fastapi_restful import Resource as fastResource
from fastapi_restful import Api as fastApi
from fastapi import FastAPI

import time

LOGGER = logging.getLogger(__name__)


for coll in list(REGISTRY._collector_to_names.keys()):  # pylint: disable=protected-access
    REGISTRY.unregister(coll)

FULL_KNN_REQUEST_TIME = Histogram("full_knn_request_time", "Time spent processing knn request")
DOWNLOAD_TIME = Histogram("download_time", "Time spent downloading an url")
TEXT_CLIP_INFERENCE_TIME = Histogram("text_clip_inference_time", "Time spent doing a text clip inference")
IMAGE_CLIP_INFERENCE_TIME = Histogram("image_clip_inference_time", "Time spent doing a image clip inference")
METADATA_GET_TIME = Histogram("metadata_get_time", "Time spent retrieving metadata")
KNN_INDEX_TIME = Histogram("knn_index_time", "Time spent doing a knn on the index")
DEDUP_TIME = Histogram("dedup_time", "Time spent deduping")
SAFETY_TIME = Histogram("safety_time", "Time spent doing a safety inference")
IMAGE_PREPRO_TIME = Histogram("image_prepro_time", "Time spent doing the image preprocessing")
TEXT_PREPRO_TIME = Histogram("text_prepro_time", "Time spent doing the text preprocessing")


def metric_to_average(metric):
    metric_data = metric.collect()[0]
    metric_name = metric_data.name
    metric_description = metric_data.documentation
    samples = metric_data.samples
    metric_sum = [sample.value for sample in samples if sample.name == metric_name + "_sum"][0]
    metric_count = [sample.value for sample in samples if sample.name == metric_name + "_count"][0]
    if metric_count == 0:
        return metric_name, metric_description, 0, 0.0
    return metric_name, metric_description, metric_count, 1.0 * metric_sum / metric_count


def convert_metadata_to_base64(meta):
    """
    Converts the image at a path to the Base64 representation and sets the Base64 string to the `image`
    key in the metadata dictionary.
    If there is no `image_path` key present in the metadata dictionary, the function will have no effect.
    """
    if meta is not None and "image_path" in meta:
        path = meta["image_path"]
        if os.path.exists(path):
            img = Image.open(path)
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            meta["image"] = img_str


class Health(Resource):
    def get(self):
        return "ok"


class MetricsSummary(Resource):
    """
    metrics endpoint for prometheus
    """

    def get(self):
        """define the metric endpoint get"""
        _, _, full_knn_count, full_knn_avg = metric_to_average(FULL_KNN_REQUEST_TIME)
        if full_knn_count == 0:
            s = "No request yet, go do some"
        else:
            sub_metrics = sorted(
                [
                    (name, description, metric_count, avg, avg / full_knn_avg)
                    for (name, description, metric_count, avg) in [
                        metric_to_average(metric)
                        for metric in [
                            DOWNLOAD_TIME,
                            TEXT_CLIP_INFERENCE_TIME,
                            IMAGE_CLIP_INFERENCE_TIME,
                            METADATA_GET_TIME,
                            KNN_INDEX_TIME,
                            DEDUP_TIME,
                            SAFETY_TIME,
                            IMAGE_PREPRO_TIME,
                            TEXT_PREPRO_TIME,
                        ]
                    ]
                ],
                key=lambda e: -e[3],
            )

            sub_metrics_strings = [
                (name, description, int(metric_count), f"{avg:0.4f}s", f"{proportion*100:0.1f}%")
                for name, description, metric_count, avg, proportion in sub_metrics
            ]

            s = ""
            s += (
                f"Among {full_knn_count} calls to the knn end point with an average latency of {full_knn_avg:0.4f}s "
                + "per request, the step costs are (in order): \n\n"
            )
            df = pd.DataFrame(
                data=sub_metrics_strings, columns=("name", "description", "calls", "average", "proportion")
            )
            s += df.to_string()

        response = make_response(s, 200)
        response.mimetype = "text/plain"
        return response

@DOWNLOAD_TIME.time()
def download_image(url):
    urllib_request = urllib.request.Request(
        url,
        data=None,
        headers={"User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:72.0) Gecko/20100101 Firefox/72.0"},
    )
    with urllib.request.urlopen(urllib_request, timeout=10) as r:
        img_stream = io.BytesIO(r.read())
    return img_stream

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)

def meta_to_dict(meta):
    output = {}
    for k, v in meta.items():
        if isinstance(v, bytes):
            v = v.decode()
        elif type(v).__module__ == np.__name__:
            v = v.item()
        output[k] = v
    return output


class ParquetMetadataProvider:
    """The parquet metadata provider provides metadata from contiguous ids using parquet"""

    def __init__(self, parquet_folder):
        data_dir = Path(parquet_folder)
        self.metadata_df = pd.concat(
            pd.read_parquet(parquet_file) for parquet_file in sorted(data_dir.glob("*.parquet"))
        )
        #print("Metadata-df of Parquet init: {0}".format(self.metadata_df))

    def get(self, ids, cols=None):
        if cols is None:
            cols = self.metadata_df.columns.tolist()
        else:
            cols = list(set(self.metadata_df.columns.tolist()) & set(cols))
            #print("return cols of parquet meta data: {0}".format(cols))
        return [self.metadata_df[i : (i + 1)][cols].to_dict(orient="records")[0] for i in ids]


def parquet_to_hdf5(parquet_folder, output_hdf5_file, columns_to_return):
    """this convert a collection of parquet file to an hdf5 file"""
    f = h5py.File(output_hdf5_file, "w")
    data_dir = Path(parquet_folder)
    ds = f.create_group("dataset")
    for parquet_files in tqdm(sorted(data_dir.glob("*.parquet"))):
        df = pd.read_parquet(parquet_files)
        for k in df.keys():
            if k not in columns_to_return:
                continue
            col = df[k]
            if col.dtype in ("float64", "float32"):
                col = col.fillna(0.0)
            if col.dtype in ("int64", "int32"):
                col = col.fillna(0)
            if col.dtype == "object":
                col = col.fillna("")
                col = col.str.replace("\x00", "", regex=False)
            z = col.to_numpy()
            if k not in ds:
                ds.create_dataset(k, data=z, maxshape=(None,), compression="gzip")
            else:
                prevlen = len(ds[k])
                ds[k].resize((prevlen + len(z),))
                ds[k][prevlen:] = z

    del ds
    f.close()


class Hdf5MetadataProvider:
    """The hdf5 metadata provider provides metadata from contiguous ids using hdf5"""

    def __init__(self, hdf5_file):
        f = h5py.File(hdf5_file, "r")
        self.ds = f["dataset"]

    def get(self, ids, cols=None):
        """implement the get method from the hdf5 metadata provide, get metadata from ids"""
        items = [{} for _ in range(len(ids))]
        if cols is None:
            cols = self.ds.keys()
        else:
            cols = list(self.ds.keys() & set(cols))
        for k in cols:
            for i, e in enumerate(ids):
                items[i][k] = self.ds[k][e]
        return items


def load_index(path, enable_faiss_memory_mapping):
    if enable_faiss_memory_mapping:
        if os.path.isdir(path):
            return faiss.read_index(path + "/populated.index", faiss.IO_FLAG_ONDISK_SAME_DIR)
        else:
            return faiss.read_index(path, faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
    else:
        return faiss.read_index(path)


class ArrowMetadataProvider:
    """The arrow metadata provider provides metadata from contiguous ids using arrow"""

    def __init__(self, arrow_folder):
        arrow_files = [str(a) for a in sorted(Path(arrow_folder).glob("**/*")) if a.is_file()]
        self.table = pa.concat_tables(
            [pa.ipc.RecordBatchFileReader(pa.memory_map(arrow_file, "r")).read_all() for arrow_file in arrow_files]
        )

    def get(self, ids, cols=None):
        """implement the get method from the arrow metadata provide, get metadata from ids"""
        if cols is None:
            cols = self.table.schema.names
        else:
            cols = list(set(self.table.schema.names) & set(cols))
        t = pa.concat_tables([self.table[i : i + 1] for i in ids])
        return t.select(cols).to_pandas().to_dict("records")


def load_metadata_provider(
    indice_folder, enable_hdf5, reorder_metadata_by_ivf_index, image_index, columns_to_return, use_arrow
):
    """load the metadata provider"""
    parquet_folder = indice_folder + "/metadata"
    ivf_old_to_new_mapping = None
    if use_arrow:
        #print("Use-Arrow enabled")
        mmap_folder = parquet_folder
        metadata_provider = ArrowMetadataProvider(mmap_folder)
    elif enable_hdf5:
        #print("Enable-hdf5 and do not use arrow.")
        hdf5_path = None
        if reorder_metadata_by_ivf_index:
            hdf5_path = indice_folder + "/metadata_reordered.hdf5"
            ivf_old_to_new_mapping_path = indice_folder + "/ivf_old_to_new_mapping.npy"
            if not os.path.exists(ivf_old_to_new_mapping_path):
                ivf_old_to_new_mapping = get_old_to_new_mapping(image_index)
                ivf_old_to_new_mapping_write = np.memmap(
                    ivf_old_to_new_mapping_path, dtype="int64", mode="write", shape=ivf_old_to_new_mapping.shape
                )
                ivf_old_to_new_mapping_write[:] = ivf_old_to_new_mapping
                del ivf_old_to_new_mapping_write
                del ivf_old_to_new_mapping
            ivf_old_to_new_mapping = np.memmap(ivf_old_to_new_mapping_path, dtype="int64", mode="r")
            if not os.path.exists(hdf5_path):
                with tempfile.TemporaryDirectory() as tmpdir:
                    re_order_parquet(image_index, parquet_folder, str(tmpdir), columns_to_return)
                    external_sort_parquet(Hdf5Sink(hdf5_path, columns_to_return), str(tmpdir))
        else:
            hdf5_path = indice_folder + "/metadata.hdf5"
            if not os.path.exists(hdf5_path):
                parquet_to_hdf5(parquet_folder, hdf5_path, columns_to_return)
        metadata_provider = Hdf5MetadataProvider(hdf5_path)
    else:
        #print("Use-arrow disabled and hdf5 disabled.")
        metadata_provider = ParquetMetadataProvider(parquet_folder)

    return metadata_provider, ivf_old_to_new_mapping


def get_cache_folder(clip_model):
    """get cache folder for given clip model"""
    from os.path import expanduser  # pylint: disable=import-outside-toplevel

    home = expanduser("~")

    cache_folder = home + "/.cache/clip_retrieval/" + clip_model.replace("/", "_")

    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder, exist_ok=True)

    return cache_folder


# needs to do this at load time
@lru_cache(maxsize=None)
def get_aesthetic_embedding(model_type):
    """get aesthetic embedding"""
    if model_type == "ViT-B/32":
        model_type = "vit_b_32"
    elif model_type == "ViT-L/14":
        model_type = "vit_l_14"
    else:
        raise ValueError(f"Aesthetic embedding for {model_type} not available.")

    fs, _ = fsspec.core.url_to_fs(
        f"https://github.com/LAION-AI/aesthetic-predictor/blob/main/{model_type}_embeddings/rating0.npy?raw=true"
    )
    embs = {}
    with ThreadPool(10) as pool:

        def get(k):
            with fs.open(
                f"https://github.com/LAION-AI/aesthetic-predictor/blob/main/{model_type}_embeddings/rating{k}.npy?raw=true",
                "rb",
            ) as f:
                embs[k] = np.load(f)

        for _ in pool.imap_unordered(get, range(10)):
            pass
    return embs


@lru_cache(maxsize=None)
def load_violence_detector(clip_model):
    """load violence detector for this clip model"""
    from urllib.request import urlretrieve  #  pylint: disable=import-outside-toplevel

    cache_folder = get_cache_folder(clip_model)
    root_url = "https://github.com/LAION-AI/CLIP-based-NSFW-Detector/raw/main"

    if clip_model == "ViT-L/14":
        name = "violence_detection_vit_l_14.npy"
    elif clip_model == "ViT-B/32":
        name = "violence_detection_vit_b_32.npy"
    else:
        raise ValueError(f"Violence detector for {clip_model} not available.")

    url_model = root_url + "/" + name
    prompt_file = cache_folder + "/" + name

    if not os.path.exists(prompt_file):
        urlretrieve(url_model, prompt_file)

    prompts = np.load(prompt_file)
    return prompts


@lru_cache(maxsize=None)
def load_safety_model(clip_model):
    """load the safety model"""
    import autokeras as ak  # pylint: disable=import-outside-toplevel
    from tensorflow.keras.models import load_model  # pylint: disable=import-outside-toplevel
    from clip_retrieval.h14_nsfw_model import H14_NSFW_Detector  # pylint: disable=import-outside-toplevel

    cache_folder = get_cache_folder(clip_model)

    if clip_model == "ViT-L/14":
        model_dir = cache_folder + "/clip_autokeras_binary_nsfw"
        dim = 768
    elif clip_model == "ViT-B/32":
        model_dir = cache_folder + "/clip_autokeras_nsfw_b32"
        dim = 512
    elif clip_model == "open_clip:ViT-H-14":
        return H14_NSFW_Detector()
    else:
        raise ValueError(f"Safety model for {clip_model} not available.")
    if not os.path.exists(model_dir):
        os.makedirs(cache_folder, exist_ok=True)

        from urllib.request import urlretrieve  # pylint: disable=import-outside-toplevel

        path_to_zip_file = cache_folder + "/clip_autokeras_binary_nsfw.zip"
        if clip_model == "ViT-L/14":
            url_model = "https://raw.githubusercontent.com/LAION-AI/CLIP-based-NSFW-Detector/main/clip_autokeras_binary_nsfw.zip"
        elif clip_model == "ViT-B/32":
            url_model = (
                "https://raw.githubusercontent.com/LAION-AI/CLIP-based-NSFW-Detector/main/clip_autokeras_nsfw_b32.zip"
            )
        else:
            raise ValueError("Unknown model {}".format(clip_model))  # pylint: disable=consider-using-f-string
        urlretrieve(url_model, path_to_zip_file)
        import zipfile  # pylint: disable=import-outside-toplevel

        with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
            zip_ref.extractall(cache_folder)

    loaded_model = load_model(model_dir, custom_objects=ak.CUSTOM_OBJECTS)
    loaded_model.predict(np.random.rand(10**3, dim).astype("float32"), batch_size=10**3)

    return loaded_model


@dataclass
class ClipResource:
    """the resource for clip : model, index, options"""

    device: str
    model: Any
    preprocess: Callable
    tokenizer: Callable
    model_txt_mclip: Any
    safety_model: Any
    violence_detector: Any
    metadata_provider: Any
    image_index: Any
    text_index: Any
    ivf_old_to_new_mapping: Any
    columns_to_return: List[str]
    metadata_is_ordered_by_ivf: bool
    aesthetic_embeddings: Any


@dataclass
class ClipOptions:
    """the options for clip"""

    indice_folder: str
    clip_model: str
    enable_hdf5: bool
    enable_faiss_memory_mapping: bool
    columns_to_return: List[str]
    reorder_metadata_by_ivf_index: bool
    enable_mclip_option: bool
    use_jit: bool
    use_arrow: bool
    provide_safety_model: bool
    provide_violence_detector: bool
    provide_aesthetic_embeddings: bool


def dict_to_clip_options(d, clip_options):
    return ClipOptions(
        indice_folder=d["indice_folder"] if "indice_folder" in d else clip_options.indice_folder,
        clip_model=d["clip_model"] if "clip_model" in d else clip_options.clip_model,
        enable_hdf5=d["enable_hdf5"] if "enable_hdf5" in d else clip_options.enable_hdf5,
        enable_faiss_memory_mapping=d["enable_faiss_memory_mapping"]
        if "enable_faiss_memory_mapping" in d
        else clip_options.enable_faiss_memory_mapping,
        columns_to_return=d["columns_to_return"] if "columns_to_return" in d else clip_options.columns_to_return,
        reorder_metadata_by_ivf_index=d["reorder_metadata_by_ivf_index"]
        if "reorder_metadata_by_ivf_index" in d
        else clip_options.reorder_metadata_by_ivf_index,
        enable_mclip_option=d["enable_mclip_option"]
        if "enable_mclip_option" in d
        else clip_options.enable_mclip_option,
        use_jit=d["use_jit"] if "use_jit" in d else clip_options.use_jit,
        use_arrow=d["use_arrow"] if "use_arrow" in d else clip_options.use_arrow,
        provide_safety_model=d["provide_safety_model"]
        if "provide_safety_model" in d
        else clip_options.provide_safety_model,
        provide_violence_detector=d["provide_violence_detector"]
        if "provide_violence_detector" in d
        else clip_options.provide_violence_detector,
        provide_aesthetic_embeddings=d["provide_aesthetic_embeddings"]
        if "provide_aesthetic_embeddings" in d
        else clip_options.provide_aesthetic_embeddings,
    )


@lru_cache(maxsize=None)
def load_mclip(clip_model):
    """load the mclip model"""
    from multilingual_clip import pt_multilingual_clip  # pylint: disable=import-outside-toplevel
    import transformers  # pylint: disable=import-outside-toplevel
    import torch  # pylint: disable=import-outside-toplevel

    if clip_model == "ViT-L/14":
        model_name = "M-CLIP/XLM-Roberta-Large-Vit-L-14"
    elif clip_model == "ViT-B/32":
        model_name = "M-CLIP/XLM-Roberta-Large-Vit-B-32"
    else:
        raise ValueError(f"Multi-lingual version of {clip_model} not available.")

    model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(model_name)
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    def encode_texts(text):
        with torch.no_grad():
            return model.forward([text], tokenizer)[0].detach().cpu().numpy()

    model_txt_mclip = encode_texts
    return model_txt_mclip


def load_clip_index(clip_options):
    """load the clip index"""
    import torch  # pylint: disable=import-outside-toplevel
    from clip_retrieval.load_clip import load_clip, get_tokenizer  # pylint: disable=import-outside-toplevel
    
    #load_clip_start = time.perf_counter()
    #start_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = load_clip(clip_options.clip_model, use_jit=clip_options.use_jit, device=device)
    #load_clip_end = time.perf_counter()
    tokenizer = get_tokenizer(clip_options.clip_model)
    #load_mclip_start = time.perf_counter()
    if clip_options.enable_mclip_option:
        model_txt_mclip = load_mclip(clip_options.clip_model)
    else:
        model_txt_mclip = None
    #load_clip_end = time.perf_counter()
    #safety_model_start = time.perf_counter()
    safety_model = load_safety_model(clip_options.clip_model) if clip_options.provide_safety_model else None
    #safety_model_end = time.perf_counter()    
    violence_detector = (
        load_violence_detector(clip_options.clip_model) if clip_options.provide_violence_detector else None
    )
    aesthetic_embeddings = (
        get_aesthetic_embedding(clip_options.clip_model) if clip_options.provide_aesthetic_embeddings else None
    )

    image_present = os.path.exists(clip_options.indice_folder + "/image.index")
    text_present = os.path.exists(clip_options.indice_folder + "/text.index")

    LOGGER.info("loading indices...")
    image_index = (
        load_index(clip_options.indice_folder + "/image.index", clip_options.enable_faiss_memory_mapping)
        if image_present
        else None
    )
    text_index = (
        load_index(clip_options.indice_folder + "/text.index", clip_options.enable_faiss_memory_mapping)
        if text_present
        else None
    )

    LOGGER.info("loading metadata...")

    metadata_provider, ivf_old_to_new_mapping = load_metadata_provider(
        clip_options.indice_folder,
        clip_options.enable_hdf5,
        clip_options.reorder_metadata_by_ivf_index,
        image_index,
        clip_options.columns_to_return,
        clip_options.use_arrow,
    )

    return ClipResource(
        device=device,
        model=model,
        preprocess=preprocess,
        tokenizer=tokenizer,
        model_txt_mclip=model_txt_mclip,
        safety_model=safety_model,
        violence_detector=violence_detector,
        metadata_provider=metadata_provider,
        image_index=image_index,
        text_index=text_index,
        ivf_old_to_new_mapping=ivf_old_to_new_mapping if clip_options.reorder_metadata_by_ivf_index else None,
        columns_to_return=clip_options.columns_to_return,
        metadata_is_ordered_by_ivf=clip_options.reorder_metadata_by_ivf_index,
        aesthetic_embeddings=aesthetic_embeddings,
    )


def load_clip_indices(
    indices_paths,
    clip_options,
) -> Dict[str, ClipResource]:
    """This load clips indices from disk"""
    LOGGER.info("loading clip...")

    with open(indices_paths, "r", encoding="utf-8") as f:
        indices = json.load(f)

    clip_resources = {}

    for name, indice_value in indices.items():
        # if indice_folder is a string
        if isinstance(indice_value, str):
            #print("Indice_value is str: {}".format(indice_value))
            clip_options = dict_to_clip_options({"indice_folder": indice_value}, clip_options)
        elif isinstance(indice_value, dict):
            #print("Indice_value is dict")
            clip_options = dict_to_clip_options(indice_value, clip_options)
        else:
            raise ValueError("Unknown type for indice_folder")
        clip_resources[name] = load_clip_index(clip_options)
        #print("Name {0} add to clip resources".format(name))
    #print("Clip option: {0}".format(clip_options))
    return clip_resources

from fastapi import Depends, FastAPI, Header, HTTPException, status, Security, Form
#from fastapi.security import OAuth2PasswordBearer
from fastapi.security.api_key import APIKeyHeader, APIKeyQuery
#oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")  # use token authentication

api_key_header = APIKeyHeader(name="aizip-token", auto_error=False)
api_key_query = APIKeyQuery(name="aizip-token", auto_error=False)
#api_keys = ["akljnv13bvi2vfo0b0bw"]
    #def api_key_auth(self, api_key: str = Depends(oauth2_scheme)):
    #    if api_key not in self.api_keys:
    #        raise HTTPException(
    #            status_code=status.HTTP_401_UNAUTHORIZED,
    #            detail="Forbidden"
    #        )

def get_api_key(
    api_keys_allowed: list=["akljnv13bvi2vfo0b0bw"],
    api_key_header: str = Security(api_key_header)
    #api_key_query: str= Security(api_key_query)
):
    print("Received api key: {}".format(api_key_header))
    if api_key_header not in api_keys_allowed:
        print("Key invalid.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Forbidden"
        )
    else:
        print("Key valid.")
        return api_key_header

def form_body(cls):
    cls.__signature__ = cls.__signature__.replace(
        parameters=[
            arg.replace(default=Form(...))
            for arg in cls.__signature__.parameters.values()
        ]
    )
    return cls

class KnnServiceFastApi(fastResource):
    """the knn service provides nearest neighbors given text or image"""

    def __init__(self, **kwargs):
        super().__init__()
        self.clip_resources = kwargs["clip_resources"]

    def compute_query(
        self,
        clip_resource,
        text_input,
        image_input,
        image_url_input,
        embedding_input,
        use_mclip,
        aesthetic_score,
        aesthetic_weight,
    ):
        """compute the query embedding"""
        import torch  # pylint: disable=import-outside-toplevel

        if text_input is not None and text_input != "":
            if use_mclip:
                with TEXT_CLIP_INFERENCE_TIME.time():
                    query = normalized(clip_resource.model_txt_mclip(text_input))
            else:
                with TEXT_PREPRO_TIME.time():
                    text = clip_resource.tokenizer([text_input]).to(clip_resource.device)
                with TEXT_CLIP_INFERENCE_TIME.time():
                    with torch.no_grad():
                        text_features = clip_resource.model.encode_text(text)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    query = text_features.cpu().to(torch.float32).detach().numpy()
        elif image_input is not None or image_url_input is not None:
            if image_input is not None:
                binary_data = base64.b64decode(image_input)
                img_data = BytesIO(binary_data)
            elif image_url_input is not None:
                img_data = download_image(image_url_input)
            with IMAGE_PREPRO_TIME.time():
                img = Image.open(img_data)
                prepro = clip_resource.preprocess(img).unsqueeze(0).to(clip_resource.device)
            with IMAGE_CLIP_INFERENCE_TIME.time():
                with torch.no_grad():
                    image_features = clip_resource.model.encode_image(prepro)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                query = image_features.cpu().to(torch.float32).detach().numpy()
                # print("Compute-Query returns query with shape: {}".format(query.shape))
        elif embedding_input is not None:
            query = np.expand_dims(np.array(embedding_input).astype("float32"), 0)

        if clip_resource.aesthetic_embeddings is not None and aesthetic_score is not None:
            aesthetic_embedding = clip_resource.aesthetic_embeddings[aesthetic_score]
            query = query + aesthetic_embedding * aesthetic_weight
            query = query / np.linalg.norm(query)

        return query

    def hash_based_dedup(self, embeddings):
        """deduplicate embeddings based on their hash"""
        seen_hashes = set()
        to_remove = []
        for i, embedding in enumerate(embeddings):
            h = hash(np.round(embedding, 2).tobytes())
            if h in seen_hashes:
                to_remove.append(i)
                continue
            seen_hashes.add(h)

        return to_remove

    def connected_components(self, neighbors):
        """find connected components in the graph"""
        seen = set()

        def component(node):
            r = []
            nodes = set([node])
            while nodes:
                node = nodes.pop()
                seen.add(node)
                nodes |= set(neighbors[node]) - seen
                r.append(node)
            return r

        u = []
        for node in neighbors:
            if node not in seen:
                u.append(component(node))
        return u

    def get_non_uniques(self, embeddings, threshold=0.94):
        """find non-unique embeddings"""
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)  # pylint: disable=no-value-for-parameter
        l, _, I = index.range_search(embeddings, threshold)  # pylint: disable=no-value-for-parameter,invalid-name

        same_mapping = defaultdict(list)

        # https://github.com/facebookresearch/faiss/wiki/Special-operations-on-indexes#range-search
        for i in range(embeddings.shape[0]):
            for j in I[l[i] : l[i + 1]]:
                same_mapping[int(i)].append(int(j))

        groups = self.connected_components(same_mapping)
        non_uniques = set()
        for g in groups:
            for e in g[1:]:
                non_uniques.add(e)

        return list(non_uniques)

    def connected_components_dedup(self, embeddings):
        non_uniques = self.get_non_uniques(embeddings)
        return non_uniques

    def get_unsafe_items(self, safety_model, embeddings, threshold=0.5):
        """find unsafe embeddings"""
        nsfw_values = safety_model.predict(embeddings, batch_size=embeddings.shape[0])
        x = np.array([e[0] for e in nsfw_values])
        return np.where(x > threshold)[0]

    def get_violent_items(self, safety_prompts, embeddings):
        safety_predictions = np.einsum("ij,kj->ik", embeddings, safety_prompts)
        safety_results = np.argmax(safety_predictions, axis=1)
        return np.where(safety_results == 1)[0]

    def post_filter(
        self, safety_model, embeddings, deduplicate, use_safety_model, use_violence_detector, violence_detector
    ):
        """post filter results : dedup, safety, violence"""
        to_remove = set()
        if deduplicate:
            with DEDUP_TIME.time():
                to_remove = set(self.connected_components_dedup(embeddings))

        if use_violence_detector and violence_detector is not None:
            to_remove |= set(self.get_violent_items(violence_detector, embeddings))
        if use_safety_model and safety_model is not None:
            with SAFETY_TIME.time():
                to_remove |= set(self.get_unsafe_items(safety_model, embeddings))

        return to_remove

    def knn_search(
        self, queries, modality, num_result_ids, clip_resource, deduplicate, use_safety_model, use_violence_detector
    ):
        """compute the knn search"""

        image_index = clip_resource.image_index
        text_index = clip_resource.text_index
        if clip_resource.metadata_is_ordered_by_ivf:
            ivf_old_to_new_mapping = clip_resource.ivf_old_to_new_mapping

        index = image_index if modality == "image" else text_index
        #print("knn search index: {}".format(index))

        with KNN_INDEX_TIME.time():
            if clip_resource.metadata_is_ordered_by_ivf:
                previous_nprobe = faiss.extract_index_ivf(index).nprobe
                if num_result_ids >= 100000:
                    nprobe = math.ceil(num_result_ids / 3000)
                    params = faiss.ParameterSpace()
                    params.set_index_parameters(index, f"nprobe={nprobe},efSearch={nprobe*2},ht={2048}")
            distances, indices, embeddings = index.search_and_reconstruct(queries, num_result_ids)
            if clip_resource.metadata_is_ordered_by_ivf:
                for idx, _ in enumerate(indices):
                    results[idx] = np.take(ivf_old_to_new_mapping, indices[idx])
            else:
                results = indices
            if clip_resource.metadata_is_ordered_by_ivf:
                params = faiss.ParameterSpace()
                params.set_index_parameters(index, f"nprobe={previous_nprobe},efSearch={previous_nprobe*2},ht={2048}")
        
            distance_outputs = []
            indices_outputs = []
            #print("knn search raw distances length : {}, indicies length: {}".format(len(distances), len(indices)))
            for idx in range(0, len(results)):
                #print("Looping in raw search results for index: {}".format(idx))
                nb_results = np.where(results[idx] == -1)[0]

                if len(nb_results) > 0:
                    nb_results = nb_results[0]
                else:
                    nb_results = len(results[idx])
                result_indices = results[idx][:nb_results]
                result_distances = distances[idx][:nb_results]
                result_embeddings = embeddings[idx][:nb_results]
                result_embeddings = normalized(result_embeddings)
                local_indices_to_remove = self.post_filter(
                    clip_resource.safety_model,
                    result_embeddings,
                    deduplicate,
                    use_safety_model,
                    use_violence_detector,
                    clip_resource.violence_detector,
                )
                indices_to_remove = set()
                #print("knn search to be removed indices len: {}".format(len(local_indices_to_remove)))
                for local_index in local_indices_to_remove:
                    indices_to_remove.add(result_indices[local_index])
                indices_filtered = []
                distances_filtered = []
                for ind, distance in zip(result_indices, result_distances):
                    if ind not in indices_to_remove:
                        indices_to_remove.add(ind)
                        indices_filtered.append(ind)
                        distances_filtered.append(distance)
                distance_outputs.append(distances_filtered)
                indices_outputs.append(indices_filtered)

            return distance_outputs, indices_outputs

    def map_to_metadata(self, indices, distances, num_images, metadata_provider, columns_to_return):
        """map the indices to the metadata"""
        #print("Num of imgs for meta data: {}".format(num_images))
        results = []
        with METADATA_GET_TIME.time():
            metas = metadata_provider.get(indices[:num_images], columns_to_return)
        for key, (d, i) in enumerate(zip(distances, indices)):
            output = {}
            meta = None if key + 1 > len(metas) else metas[key]
            print("Meta for image with id {}:{}".format(i.item(), meta))
            convert_metadata_to_base64(meta)
            if meta is not None:
                output.update(meta_to_dict(meta))
            output["id"] = i.item()
            output["similarity"] = d.item()
            results.append(output)

        return results

    def create_img_list(self, folder_path):
        """use base64 to encode images from given folder, accepted format .jpg, .png"""
        import base64
        encoded_img_list = []
        img_format = ["jpg", "png"]
        if not os.path.isdir(folder_path):
            print("The folder path does not exist")
            return encoded_img_list
        else:
            for file in os.listdir(folder_path):
                filename = os.fsdecode(file)
                file_path = os.path.join(folder_path, filename)
                #print("file: {0}, filename: {1}".format(file, file_path))  ## use for debug
                if filename.rsplit(".",1)[1] in img_format:
                    with open(file_path, "rb") as img:
                        encoded_img = base64.b64encode(img.read())
                        encoded_img_list.append(encoded_img)
        return encoded_img_list

    # TODO @Junjie, add interfaces: image_input/image_url_input/embedding_input
    def multi_img_query(
        self,
        image_folder=None,
        image_urls=None,
        embeddings=None,
        model="ViT-L/14",
        modality="image",
        num_images=100,
        num_result_ids=100,
        indice_name=None,
        deduplicate=True,
        use_safety_model=False,
        use_violence_detector=False,
        aesthetic_score=None,
        aesthetic_weight=None,
    ):
        """implement the querying functionality of the knn service: from text and image to nearest neighbors"""
        if image_folder is None and image_urls is None and embeddings is None:
            raise ValueError("must fill embeddings or an img folder of a list of urls as input")
        
        start_knn_query = time.perf_counter()
        if indice_name is None:
            indice_name = next(iter(self.clip_resources.keys()))
            print("use first key of indices as default: {0}".format(indice_name))

        clip_resource = self.clip_resources[indice_name]

        """check model, drop is unknown"""
        if model == "ViT-B/32":
            queries = np.empty([1, 512])
        elif model == "ViT-L/14":
            queries = np.empty([1, 768])
        else:
            print("model unknown")
            return []

        """check input image folder"""
        if image_folder is not None:
            image_inputs = self.create_img_list(image_folder)
            if image_inputs is None:
                print("Empty img input list, check if the input folders are empty.")
                return []
            elif len(image_inputs)==1:
                print("Only one image detected, use single query instead.")
                return self.query(
                    image_input=image_inputs[0],
                    modality=modality,
                    num_images=num_images,
                    num_result_ids=num_result_ids,
                    indice_name=indice_name,
                    deduplicate=deduplicate,
                    use_mclip=False,
                    use_safety_model=use_safety_model,
                    use_violence_detector=use_violence_detector,
                    aesthetic_score=aesthetic_score,
                    aesthetic_weight=aesthetic_weight,
                    )
            else:
                import torch  # pylint: disable=import-outside-toplevel
                for idx, img in enumerate(image_inputs):
                    binary_data = base64.b64decode(img)
                    img_data = BytesIO(binary_data)

                    with IMAGE_PREPRO_TIME.time():
                        img = Image.open(img_data)
                        prepro = clip_resource.preprocess(img).unsqueeze(0).to(clip_resource.device)
                    with IMAGE_CLIP_INFERENCE_TIME.time():
                        with torch.no_grad():
                            image_features = clip_resource.model.encode_image(prepro)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    query = image_features.cpu().detach().numpy().astype("float32")

                    if clip_resource.aesthetic_embeddings is not None and aesthetic_score is not None:
                        aesthetic_embedding = clip_resource.aesthetic_embeddings[aesthetic_score]
                        query = query + aesthetic_embedding * aesthetic_weight
                        query = query / np.linalg.norm(query)

                    if idx==0:
                        queries = query
                    else:
                        queries = np.append(queries, query, axis=0)

                print("Shape of query: {0}".format(queries.shape))
                    
                distances_arr, indices_arr = self.knn_search(
                    queries,
                    modality=modality,
                    num_result_ids=num_result_ids,
                    clip_resource=clip_resource,
                    deduplicate=deduplicate,
                    use_safety_model=use_safety_model,
                    use_violence_detector=use_violence_detector,
                )

                results_arr = []
                for idx in range(0, len(distances_arr)):
                    if len(distances_arr[idx]) == 0 or len(indices_arr[idx]) == 0:
                        results_arr.append([])
                        print("For idx: {0} in raw search there is either no distance or no index returned.".format(idx))
                    else:
                        results = self.map_to_metadata(
                            indices_arr[idx], distances_arr[idx], num_images, clip_resource.metadata_provider, clip_resource.columns_to_return
                        )
                        results_arr.append(results)

                end_knn_query = time.perf_counter()
                LOGGER.info(f'Total duration for query: {end_knn_query-start_knn_query}')
                return results_arr
        elif image_urls is not None:
            image_inputs = image_urls
            #ToDo add implementation for image_urls
            return []
        elif embeddings is not None:
            image_inputs = embeddings 
            #ToDo add implmentation for embeddings
            return []      

    def multi_query(
        self,
        text_input=None,
        image_input_list=None,
        image_url_input_list=None,
        embedding_input_list=None,
        modality="image",
        num_images=100,
        num_result_ids=100,
        indice_name=None,
        use_mclip=False,
        deduplicate=True,
        use_safety_model=False,
        use_violence_detector=False,
        aesthetic_score=None,
        aesthetic_weight=None,  
    ):
        """
        central interface for multiple inputs, with compatibility to single input
        """
        text_result = []
        if text_input is None and image_input_list is None and image_url_input_list is None and embedding_input_list is None:
            raise ValueError("must fill one of text, image and image url input")
        
        if text_input is not None and text_input != "":
            text_result = self.query(
                text_input=text_input,
                modality=modality,
                num_images=num_images,
                num_result_ids=num_result_ids,
                indice_name=indice_name,
                deduplicate=deduplicate,
                use_mclip=False,
                use_safety_model=use_safety_model,
                use_violence_detector=use_violence_detector,
                aesthetic_score=aesthetic_score,
                aesthetic_weight=aesthetic_weight,
                )
        elif image_input_list is not None or image_url_input_list is not None or embedding_input_list is not None:
            pass

        #return results_arr
    
    def classifier_img_query(
        self,
        cls_weight_csv=None, # in default sep is empty space
        cls_weight_emb=None, # in default sep is empty space
        cls_bias_csv=None,
        cls_bias_emb=None,
        num_images=10,
        num_result_ids=10,
        indice_name=None,
        deduplicate=True,
        use_safety_model=False,
        use_violence_detector=False,
        aesthetic_score=None,
        aesthetic_weight=None,
        threshold=0.0,
    ):
        def getValidIdx(res_list, thres, bias):
            valid_idx_rev = 0
            for idx, res in enumerate(reversed(res_list),1):
                if res['similarity'] + bias >= thres:
                    break
                else:
                    valid_idx_rev = idx
            return valid_idx_rev
        
        import time
        start_t = time.perf_counter()
        if cls_weight_csv is not None:
            weight_emb = np.loadtxt(
                cls_weight_csv,
                delimiter=' '   # default delimiter is empty space
            )
            weight_emb /= np.linalg.norm(weight_emb, keepdims=True)
        elif cls_weight_emb is not None:
            weight_emb = cls_weight_emb
        else:
            print("No valid input for class weights can be found. Exit.")
            exit()
        
        if cls_bias_csv is not None:
            bias = np.loadtxt(
                cls_bias_csv,
                delimiter=' '   # default delimiter is empty space
            )
        elif cls_bias_emb is not None:
            bias = cls_bias_emb
        else:
            print("No valid input for class bias can be found. Take default value 0.")
            bias = 0

        raw_q_res = self.query(
            embedding_input=weight_emb,
            modality="image",
            num_images=num_images,
            num_result_ids=num_result_ids,
            indice_name=indice_name,
            use_mclip=False,
            deduplicate=deduplicate,
            use_safety_model=use_safety_model,
            use_violence_detector=use_violence_detector,
            aesthetic_score=aesthetic_score,
            aesthetic_weight=aesthetic_weight,  
        )

        start_filter = time.perf_counter()
        valid_idx = 0
        if len(raw_q_res) == 0:
            print("Enpty results returned...")
            return []
        elif len(raw_q_res) > 100000:
            # take half for validation
            half_idx = int((len(raw_q_res)-1)/2)
            if raw_q_res[half_idx]['similarity'] + bias >= threshold:
                valid_idx = len(raw_q_res) - getValidIdx(raw_q_res[half_idx:], threshold, bias)
            else:
                valid_idx = half_idx - getValidIdx(raw_q_res[:half_idx], threshold, bias)
        else:
            valid_idx = len(raw_q_res) - getValidIdx(raw_q_res, threshold, bias)
        print("Duration for filtering query results: {}".format(time.perf_counter()-start_filter))
        print("Total duration for query with classifier: {}".format(time.perf_counter()-start_t))
        return raw_q_res[:valid_idx]

    def query(
        self,
        text_input=None,
        image_input=None,
        image_url_input=None,
        embedding_input=None,
        modality="image",
        num_images=100,
        num_result_ids=100,
        indice_name=None,
        use_mclip=False,
        deduplicate=True,
        use_safety_model=False,
        use_violence_detector=False,
        aesthetic_score=None,
        aesthetic_weight=None,
    ):
        """implement the querying functionality of the knn service: from text and image to nearest neighbors"""
        knn_query_start = time.perf_counter()
        if text_input is None and image_input is None and image_url_input is None and embedding_input is None:
            raise ValueError("must fill one of text, image and image url input")
        if indice_name is None:
            indice_name = next(iter(self.clip_resources.keys()))

        print("clip resources keys: {}, received key: {}".format(self.clip_resources.keys(), indice_name))
        clip_resource = self.clip_resources[indice_name]
        
        query = self.compute_query(
            clip_resource=clip_resource,
            text_input=text_input,
            image_input=image_input,
            image_url_input=image_url_input,
            embedding_input=embedding_input,
            use_mclip=use_mclip,
            aesthetic_score=aesthetic_score,
            aesthetic_weight=aesthetic_weight,
        )
        distances, indices = self.knn_search(
            query,
            modality=modality,
            num_result_ids=num_result_ids,
            clip_resource=clip_resource,
            deduplicate=deduplicate,
            use_safety_model=use_safety_model,
            use_violence_detector=use_violence_detector,
        )
        if len(distances) == 0:
            return []
        #print("len of distance: {0}, len of indices: {1}".format(len(distances), len(indices)))
        results = self.map_to_metadata(
            indices[0], distances[0], num_images, clip_resource.metadata_provider, clip_resource.columns_to_return
        )
        knn_query_end = time.perf_counter()
        LOGGER.info(f'Total duration for query: {knn_query_end-knn_query_start}')

        return results

    from pydantic import BaseModel
    @form_body
    class KnnPostReq(BaseModel):
        #from typing import Union, Optional
        image: str = None
        modality: str = "image"
        num_images: int = 1
        embedding_input: list = None
        indice_name: str 
        use_mclip: bool = False
        deduplicate: bool = False
        use_safety_model: bool = False
        use_violence_detector: bool = False
        #api_key: str = Depends(get_api_key)
        #class Config:
        #    arbitrary_types_allowed = True

    @FULL_KNN_REQUEST_TIME.time()
    def post(self, request: KnnPostReq = Depends(KnnPostReq)): #, api_key=Security(get_api_key)):
        """implement the post method for knn service, parse the request and calls the query method"""
        #print("Api key used: {}".format(api_key))
        #json_data = json.loads(request.json())
        print("Request indice_name: {}".format(request.indice_name))
    
        image_input = None if request.image == "None" else request.image
        embedding_input = None if len(request.embedding_input)==1 and request.embedding_input[0]=="" else request.embedding_input
        print("Requested image is None? :{}, requested embedding is None? : {}".format(image_input==None, embedding_input==None))

        if embedding_input != None and len:
            embedding_input = embedding_input[0][3:-3]
            embedding_input_str_list = embedding_input.split(", ") # content is string form of string array, "[]" needs to be eliminated first
            embedding_input_float_list = [float(number[2:-2]) for number in embedding_input_str_list] # content is again string form of string, "'xxx'"
            embedding_input = embedding_input_float_list
        modality = request.modality
        num_images = request.num_images
        num_result_ids = request.num_images
        indice_name = request.indice_name
        use_mclip = request.use_mclip
        deduplicate = request.deduplicate
        use_safety_model = request.use_safety_model
        use_violence_detector = request.use_violence_detector
        #aesthetic_score = json_data.get("aesthetic_score", "")
        #aesthetic_score = int(aesthetic_score) if aesthetic_score != "" else None
        #aesthetic_weight = json_data.get("aesthetic_weight", "")
        #aesthetic_weight = float(aesthetic_weight) if aesthetic_weight != "" else None
        
        return self.query(
            #text_input,
            image_input=image_input,
            #image_url_input,
            embedding_input=embedding_input,
            modality=modality,
            num_images=num_images,
            num_result_ids=num_result_ids,
            indice_name=indice_name,
            use_mclip=use_mclip,
            deduplicate=deduplicate,
            use_safety_model=use_safety_model,
            use_violence_detector=use_violence_detector,
            #aesthetic_score,
            #aesthetic_weight,
        )

class MetadataServiceFastApi(fastResource):
    """The metadata service provides metadata given indices"""

    def __init__(self, **kwargs):
        super().__init__()
        self.clip_resources = kwargs["clip_resources"]

    def post(self):
        """Post the metadata"""
        json_data = request.get_json(force=True)
        ids = json_data["ids"]
        if len(ids) == 0:
            return []
        indice_name = json_data["indice_name"]
        metadata_provider = self.clip_resources[indice_name].metadata_provider
        metas = metadata_provider.get(ids, self.clip_resources[indice_name].columns_to_return)
        for meta in metas:
            convert_metadata_to_base64(meta)
        metas_with_ids = [{"id": item_id, "metadata": meta_to_dict(meta)} for item_id, meta in zip(ids, metas)]
        return metas_with_ids

class IndicesListFastApi(fastResource):
    def __init__(self, **kwargs):
        super().__init__()
        self.indices = kwargs["indices"]

    def get(self):
        return self.indices

def clip_back_fastapi(
    index_folder="",
    enable_hdf5=False,
    enable_faiss_memory_mapping=False,
    columns_to_return=None,
    reorder_metadata_by_ivf_index=False,
    default_backend=None,
    url_column="url",
    enable_mclip_option=True,
    clip_model="ViT-B/32",
    use_jit=True,
    use_arrow=False,
    provide_safety_model=False,
    provide_violence_detector=False,
    provide_aesthetic_embeddings=True,
):
    """main entry point of clip back, start the endpoints"""
    print("starting boot of clip back using fastapi.")
    columns_to_return = ["url", "image_path", "caption", "NSFW"] if columns_to_return is None else columns_to_return
    clip_resource = load_clip_index(
        clip_options=ClipOptions(
            indice_folder=index_folder,
            clip_model=clip_model,
            enable_hdf5=enable_hdf5,
            enable_faiss_memory_mapping=enable_faiss_memory_mapping,
            columns_to_return=columns_to_return,
            reorder_metadata_by_ivf_index=reorder_metadata_by_ivf_index,
            enable_mclip_option=enable_mclip_option,
            use_jit=use_jit,
            use_arrow=use_arrow,
            provide_safety_model=provide_safety_model,
            provide_violence_detector=provide_violence_detector,
            provide_aesthetic_embeddings=provide_aesthetic_embeddings,
        ),
    )
    clip_resources={}
    clip_resources[index_folder] = clip_resource
    print("indices loaded, using key {}.".format(index_folder))
    app = FastAPI(dependencies=[Security(get_api_key)])
    #app = FastAPI()
    api = fastApi(app)

    #app.add_middleware
    indices = IndicesListFastApi(indices=list([index_folder]))
    api.add_resource(indices, "/indices-list")
    metadata = MetadataServiceFastApi(clip_resources=clip_resources)
    api.add_resource(metadata, "/metadata")
    knn = KnnServiceFastApi(clip_resources=clip_resources)
    api.add_resource(knn, "/knn-service")
    return app
    #uvicorn.run(app, port=port, host=host)

if __name__ == "__main__":
    fire.Fire(clip_back)
