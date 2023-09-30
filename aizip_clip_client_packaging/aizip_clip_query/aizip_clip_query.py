import requests
#import logging
import numpy as np
import pandas as pd
import os
import base64
from datetime import datetime

class ClipBaseQuery:
    """"""
    def __init__(self, **kwargs):
        assert kwargs["host_uri"], f"'host_uri' is not given as parameter."
        self.host = kwargs["host_uri"]
        self.port = kwargs["port"]
        self.debug_print = True

        #logging.basicConfig(
        #    filename='',
        #    encoding='utf-8',
        #    level=logging.DEBUG,
        #)
        self.headers = {
            'accept': 'application/json',
            'aizip-token': 'akljnv13bvi2vfo0b0bw',
            #'Content-Type': 'application/json',
        }
        #self.__get_request("/config.json")
        self.indices = self.__get_request("/indices-list")
    
    def __get_request(self, url, headers=None):
        """"""
        total_url = self.host + ":" + self.port + url
        response = requests.get(total_url, headers=self.headers)
        if self.debug_print:
            print("Available indices: {}".format(response))
        return response.json()
    
    def post_request(self, url, payload, headers=None):
        """"""
        pass
    
    def get_index_name(self, indices_idx=0):
        indices_list = self.indices[indices_idx].strip().strip('["').strip('"]').split(",", -1) #.replace('\\\\','\\')
        index_name = indices_list[indices_idx] if indices_idx <= len(indices_list) else indices_list[0]
        if self.debug_print:
            print("index_name: {} with type: {}".format(index_name, type(index_name)))
        return index_name
    
    def __query(self, **kwargs):
        raise NotImplementedError
    
    def post_print(self, json_results):
        for idx, result in enumerate(json_results):
            if len(result) == 1:
                res_table = pd.DataFrame(
                    [(result['image_path'], result['id'], result['similarity'])],
                    columns=["image_path", "id", "similarity"],
                )
                print("====================================")
                print("======= results for image {} =======".format(idx))
                print(res_table)
            else:
                res_table = pd.DataFrame(
                    [(e['image_path'], e['id'], e['similarity']) for e in result],
                    columns=["image_path", "id", "similarity"],
                )
                print("====================================")
                print("======= results for image {} =======".format(idx))
                print(res_table)


class ClipImgQuery(ClipBaseQuery):
    """"""
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def query_for_image_dir(self, dir, num_res=1, indices_idx=0, print_res=False):
        """"""
        if len(self.indices) == 0:
            if self.debug_print:
                print("There is no valid index available on server. Return empty results.")
            return {}
        
        if os.path.exists(dir) == False:
           print("Given image folder doesn´t exist. Return empty result.")
           return {}
        # todo: extend other image formats
        from pathlib import Path
        images = Path(dir).glob("*.jpg")
        print("Accept only images with .jpg format!!")
        img_byte_list = []
        img_name_list = []
        for image in images:
            # encode image bytes with b42 first
            with open(image, "rb") as img_file:
                img_byte_list.append(self.__image_encoding(img_file.read()))
                img_name_list.append(image.name)

        return self.__query(
                images_string=str(img_byte_list),
                num_res = num_res,
                indices_idx = indices_idx,
                print_res = print_res
            )

    def query_for_single_image(self, image_url, num_res=1, indices_idx=0, print_res=False):
        """"""        
        if os.path.exists(image_url) == False:
            print("Given image path is not valid, no image can be loaded. Return emtpy results.")
            return {}
        with open(image_url, "rb") as image:
            img_byte= [self.__image_encoding(image.read())]

        return self.__query(
            images_string=str(img_byte),
            num_res = num_res,
            indices_idx = indices_idx,
            print_res = print_res
        )

    def __image_encoding(self, image, encoding="base64"):
        if encoding == "base64":
            return base64.b64encode(image)
        else:
            print("This encoding: {} is not supported yet.".format(encoding))
            return None
        
    def __query(self, **kwargs):
        assert kwargs["images_string"], f"'image_string' is not given as parameter for query in ClipImg class."
        
        total_url = self.host + ":" + self.port + "/knn-service"

        indice_name_str = self.get_index_name(kwargs["indices_idx"]) if kwargs["indices_idx"] else self.get_index_name(0)
        num_res_str = str(kwargs["num_res"]) if kwargs["num_res"] else "1"

        payload = {
            'query_images_list': (None, kwargs["images_string"]),
            'query_embeddings_list': (None, ''),
            'modality': (None, 'image'),
            'use_mclip': (None, 'false'),
            'deduplicate': (None, 'false'),
            'num_images': (None, num_res_str),
            'use_violence_detector': (None, 'false'),
            'use_safety_model': (None, 'false'),
            'indice_name': (None, indice_name_str),
        }

        query_start_time = datetime.now()
        response = requests.post(total_url, files=payload, headers=self.headers)
        query_end_time = datetime.now()
        response_json = response.json()
        if self.debug_print:
            print("Status code: {}".format(response.status_code))
        print("Total time for query: {}".format(query_end_time - query_start_time))

        if kwargs["print_res"]:
            self.post_print(response_json)
        return response_json

class ClipClassifierQuery(ClipBaseQuery):
    """"""
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def query_for_csv(self, csv_file, csv_idx=[0], num_res=1, indices_idx=0, print_res=False):        
        if os.path.exists(csv_file) == False:
           print("Given csv file doesn´t exist. Return empty result.")
           return {}
        csv_emb = self.__load_emb_from_csv(csv_file, ' ')
        if csv_emb is None:
            print("No valid embeddings can be read from given .csv file.")
            return []
        embeddings = []
        for idx in csv_idx:
            if idx < csv_emb.shape[0]:
                embeddings.append(self.__convert_emb_to_str(csv_emb[idx,:]))
        return self.__query(
            embeddings_str=str(embeddings),
            num_res=num_res,
            indices_idx=indices_idx,
            print_res=print_res,
        )        

    def query_for_single_embedding(self, embedding, num_res=1, indices_idx=0, print_res=False):
        return self.__query(
            embeddings_str = str([self.__convert_emb_to_str(embedding)]),
            num_res = num_res, 
            indices_idx = indices_idx,
            print_res=print_res,
        )
    
    def query_for_embeddings(self, embeddings, num_res=1, indices_idx=0, print_res=False):
        embeddings_list = [self.__convert_emb_to_str(embedding) for embedding in embeddings]
        return self.__query(
            embeddings_str = str(embeddings_list),
            num_res = num_res, 
            indices_idx = indices_idx,
            print_res=print_res,
        )

    def __load_emb_from_csv(self, csv_file, sep=' '):
        """"""
        return np.genfromtxt(csv_file, delimiter=sep)

    def __convert_emb_to_str(self, embedding):
        return ["%.10f" % number for number in embedding]

    def __query(self, **kwargs):
        assert kwargs["embeddings_str"], f"'embeddings_str' is not given as parameter for query in ClipImg class."
        total_url = self.host + ":" + self.port + "/knn-service"
        indice_name_str = self.get_index_name(kwargs["indices_idx"]) if kwargs["indices_idx"] else self.get_index_name(0)
        num_res_str = str(kwargs["num_res"]) if kwargs["num_res"] else "1"

        payload = {
            'query_embeddings_list': (None, kwargs["embeddings_str"]),
            'query_images_list': (None, ''),
            'modality': (None, 'image'),
            'use_mclip': (None, 'false'),
            'deduplicate': (None, 'false'),
            'num_images': (None, num_res_str),
            'use_violence_detector': (None, 'false'),
            'use_safety_model': (None, 'false'),
            'indice_name': (None, indice_name_str),
        }
        
        query_start_time = datetime.now()
        response = requests.post(total_url, files=payload, headers=self.headers)
        query_end_time = datetime.now()
        response_json = response.json()
        if self.debug_print:
            print("Status code: {}".format(response.status_code))
        print("Total time for query: {}".format(query_end_time - query_start_time))

        if kwargs["print_res"]:
            self.post_print(response_json)
        return response_json