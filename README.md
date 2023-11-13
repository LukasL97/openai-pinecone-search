# openai-pinecone-search

This repository contains the code for the blog post 
*[Answer questions about your documents with OpenAI and Pinecone](https://www.codecentric.de/wissens-hub/blog/answer-questions-about-your-documents-with-openai-and-pinecone)*.

The demo script works with documents stored in the *data* directory.
Currently the directory contains 1000 documents from the [Wikitext](https://huggingface.co/datasets/wikitext) dataset,
that can be downloaded using the *wikitext.py* script.
Additionally, I added the fake article *James Miller (Chemist)* to provide an example for a text that can not have
been seen by the OpenAI model before.


## Create the Pinecone index

```shell
python3 demo.py create_index
```

## Fill the Pinecone index with documents from *data*

```shell
python3 demo.py fill_index
```

## Answer a question using OpenAI and the Pinecone index

```shell
python3 demo.py get_answer
```
