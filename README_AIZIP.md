# Change log
## Clip inference
* extend input interface to accept a list of image folders
* adapt partition calculation in case of multiple image folders
* add gpu parallel computing based on visible devices (supports only image inputs)

## Clip backend
* extend input interface for query with image folder

# User Instructions
## scripts\embedding.py
provide different help functions to get required image folders. Provide main function to generate embeddings. Specify input datasets and output folder.
Modify input parameters for func "callinference" for direct use.

## scripts\index.py
provide main function to generate indices from given embeddings. Use --help or -h for more details

## scripts\query.py
provide main function to query with a folder of images. Use --help or -h for more details

