# Aizip ClipRetrieval 0.1.0
### November 9, 2023

## New
* Added low level http connection settings and embedded functions to get meta information of the internal clip server.

* Added class for image query. Provided following query format as interfaces:
    - single image
    - image folder

* Added class for embedding query. Provided following query format as interfaces:
    - csv file
    - single embedding
    - embeddings

* Added number of return images and choice of index as common interface parameters for queries. Both are optional query parameters and have default values.

* Added extra user options for each query interface, to allow printing out the query result and saving the query result to a html file. Saving into an extra file will be very helpful for large amount of query results and post processing. Both are optional query parameters and have default values.