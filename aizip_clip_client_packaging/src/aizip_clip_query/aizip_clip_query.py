import http.client
import logging
import json
import pandas as pd
import numpy as np
import os
import base64

class ClipBaseQuery:
    """"""
    def __init__(self, **kwargs):
        assert kwargs["host_uri"], f"'host_uri' is not given as parameter."
        self.host = kwargs["host_uri"]
        self.port = kwargs["port"]
        self.https_conn = http.client.HTTPConnection(
            host=self.host,
            port=self.port,
            #timeout=10,
        )
        self.debug_print = True

        logging.basicConfig(
            filename='',
            encoding='utf-8',
            level=logging.DEBUG,
        )
        self.__get_request("/config.json")
        self.indices = self.__get_request("/indices-list")
    
    def __get_request(self, url, headers=None):
        """"""
        self.https_conn.request("GET", url, headers)
        res =self.https_conn.getresponse()
        data =res.read().decode("utf-8")
        if self.debug_print:
            print(data)
        return data
    
    def post_request(self, url, payload, headers):
        """"""
        self.https_conn.request("POST", url, payload, headers)
        res =self.https_conn.getresponse()
        data =res.read().decode("utf-8")
        if self.debug_print:
            print(data)
        return data
    
    def get_index_name(self, indices_idx=0):
        indices_list = self.indices.strip().strip('["').strip('"]').replace('\\\\','\\').split(",", -1)
        index_name = indices_list[indices_idx] if indices_idx <= len(indices_list) else indices_list[0]
        if self.debug_print:
            print("index_name: {} with type: {}".format(index_name, type(index_name)))
        return index_name
    
    def __query(self, payload, headers):
        raise NotImplementedError

class ClipImgQuery(ClipBaseQuery):
    """"""
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def query_for_image_dir(self, dir, num_res=1, indices_idx=0):
        """"""
        if os.path.exists(dir) == False:
            return None
        # todo: extend other image formats
        from pathlib import Path
        images = Path(dir).glob("*.jpg")
        results = {}
        for image in images:
            # encode image bytes with b42 first
            with open(image, "rb") as img_file:
                res = self.__fill_and_query(self.__image_encoding(img_file), num_res, indices_idx)
                results[image] = json.loads(res)
        return results

    def query_for_single_image(self, image_url, num_res=1, indices_idx=0):
        """"""
        if os.path.exists(image_url) == False:
            return None
        with open(image_url, "rb") as image:
            res = self.__fill_and_query(self.__image_encoding(image.read()), num_res, indices_idx)
        return json.loads(res)

    def __image_encoding(self, image, encoding="base64"):
        return base64.b64encode(image).decode("utf-8")
    
    def __fill_and_query(self, image_bytes, num_res=1, indices_idx=0):
        """"""
        if len(self.indices) == 0:
            if self.debug_print:
                print("There is no valid index available on server. Return empty results.")
            return []

        payload = {
            'image': image_bytes,
            'modality': 'image',
            'num_images': num_res,
            'indice_name': self.get_index_name(indices_idx),
        }
        headers = {
            'content-type': 'application/json', 
            'Accept': 'application/json',
        }
        print("payload: {}".format(payload))
        return self.__query(payload, headers)

    def __query(self, payload, headers):
        return self.post_request("/knn-service", json.dumps(payload), headers)

class ClipClassifierQuery(ClipBaseQuery):
    """"""
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def query_for_csv(self, csv_file, csv_idx=0, num_res=1, indices_idx=0):
        embeddings = self.__load_emb_from_csv(csv_file, ' ')
        if embeddings is None:
            print("No valid embeddings can be read from given .csv file.")
        res = self.__fill_and_query(embeddings[csv_idx,:].tolist(), num_res, indices_idx)
        return json.loads(res)

    def query_for_single_embedding(self, embedding, num_res=1, indices_idx=0):
        res = self.__fill_and_query(embedding, num_res, indices_idx)
        return json.loads(res)

    def __load_emb_from_csv(self, csv_file, sep=' '):
        """"""
        return np.genfromtxt(csv_file, delimiter=sep) if os.path.exists(csv_file) == True else None

    def __fill_and_query(self, embedding, num_res=1, indices_idx=0):
        """
        """
        if len(self.indices) == 0:
            if self.debug_print:
                print("There is no valid index available on server. Return empty results.")
            return []
        payload = {
            'embedding_input': embedding,
            'modality': 'image',
            'num_images': num_res,
            'indice_name': self.get_index_name(indices_idx),
        }
        headers = {
            'content-type': 'application/json', 
            'Accept': 'application/json',
        }
        return self.__query(payload, headers)

    def __query(self, payload, headers):
        return self.post_request("/knn-service", json.dumps(payload), headers)