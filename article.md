# Answer questions about your documents with OpenAI and Pinecone

## Approach

## Implementation

### Set up OpenAI and Pinecone

In order to use OpenAI and Pinecone, we first need API access to both services.
We can create an OpenAI account at https://platform.openai.com/ and then create a new API key
at https://platform.openai.com/account/api-keys.
For Pinecone, we create an account and a new project in the Pinecone console at https://app.pinecone.io/ and then go
to the *API keys* tab to create a new API key.
The API key should use the *gcp-starter* environment.

We set up both key values in an *.env* file, which our python script will use to load the values:

```
OPENAI_API_KEY=<YOUR OPENAI API KEY>
PINECONE_API_KEY=<YOUR PINECONE API KEY>
```

We initialize both the OpenAI client and the Pinecone client in the python script:

```python
import os
import openai
import pinecone
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment='gcp-starter')
```

### Create the Pinecone index

We can create a Pinecone index either in the Pinecone console or programmatically using the client.
Here, we will do the latter to create an index named *document-search-index*.
The index can be configured with regard to different parameters, most notably the dimensions and the metric.
The dimensions specify the size of the vectors that we will store in the index.
As our embedding model will use vectors of size 1,536, we set the dimensions accordingly.
For the metric we have the choice between *cosine*, *dotproduct* and *euclidean*.
[As the OpenAI documentation recommends using cosine similarity](https://platform.openai.com/docs/guides/embeddings/which-distance-function-should-i-use),
we will use the *cosine* metric.

```python
pinecone_index_name = 'document-search-index'

def create_pinecone_index():
    pinecone.create_index(pinecone_index_name, metric='cosine', dimension=1536)


create_pinecone_index()
```

We can also configure the index with regard to the number of pods and pod type.
However, in the free tier we are limited to a single pod and pod type.
The [Pinecone documentation](https://docs.pinecone.io/docs/choosing-index-type-and-size) explains how the index can be
configured in more detail.

### Embed and store your documents in the Pinecone index

Now that we have created the Pinecone index, we can embed and store our documents in the index.
First, we need to load our documents from the disk.
In this case, we assume that the documents are stored in a directory named *data*.
The documents are loaded from the directory and returned as a list of dicts, consisting of the *title* (i.e. the
filename
without ending) and the *content*.

```python
import os

def load_documents():
    documents = []
    documents_path = 'data'
    for filename in os.listdir(documents_path):
        file_path = os.path.join(documents_path, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        documents.append({'title': filename.split('.')[0], 'content': content})
    return documents
```

Next, we need to a function to embed the content of a document using OpenAI's embedding model.
The OpenAI client offers an endpoint for that, which allows us to specify an embedding model.
We use the model *text-embedding-ada-002*, which is
[recommended by OpenAI](https://platform.openai.com/docs/guides/embeddings/embedding-models) at the time of
writing this article. The model generates embedding vectors of size 1,536.

```python
import openai.embeddings_utils

def get_embedding_vector_from_openai(text):
    return openai.embeddings_utils.get_embedding(text, engine='text-embedding-ada-002')
```

With the documents and the embedding function, we are now able to fill our Pinecone index with the embedded documents.
The *upsert* method of the Pinecone client expects a list of vectors with *id*, *values* (i.e. the actual vector), and
*metadata*.
The *id* is a unique identifier for each vector in the index and can be used to query a particular vector.
As we won't need this in our use case, we simply set a random value as *id*.
The *metadata* can be any additional information that we want to store together with the vector.
In this case, we store the title of the document as *metadata*.

```python
import time
import uuid

def fill_pinecone_index(documents):
    index = pinecone.Index(pinecone_index_name)
    for doc in documents:
        try:
            embedding_vector = get_embedding_vector_from_openai(doc['content'])
            data = pinecone.Vector(
                id=str(uuid.uuid4()),
                values=embedding_vector,
                metadata={'title': doc['title']}
            )
            index.upsert([data])
            print(f'Embedded and inserted document with title ' + doc['title'])
            time.sleep(3)
        except:
            print(f'Could not embed and insert document with title ' + doc['title'])

documents = load_documents()
fill_pinecone_index(documents)
```

You may note that we have added a `time.sleep(3)` after each embedding and insertion.
This is in order to avoid a rate limit error from OpenAI, which only allows a certain number of tokens to be embedded
per minute.
Further, the embedding model we use is currently limited to texts of up to 8,191 input tokens, which may not be
enough for all documents in *data*.
In this case, we simply skip the embedding and insertion of these document, so that not all our documents will end up
as vectors in the index.
If you have long documents with a lot of text, you may want to consider splitting them into smaller chunks and embed
those individually.

### Answer questions about the documents

To answer questions about our documents, we will find the relevant documents by querying the Pinecone index and then
use these documents and the question to create a prompt for the OpenAI chat completion endpoint, which asks the model
to answer the question based on the given text.

To retrieve the relevant documents, we simply embed the question using the same model that we used to embed the
documents.
Then, we query the index with this embedding vector, which will retrieve the top *k* similar vectors in the index.
We set *k* to 1 in this case, as we only answer the question based on a single document, but a larger value can be used
as well, to then create a prompt that asks the chat model to answer the question based on multiple texts.
We fetch the title of the document from the metadata, which will enable us to retrieve the document from the disk.

```python
def query_pinecone_index(query):
    index = pinecone.Index(pinecone_index_name)
    query_embedding_vector = get_embedding_vector_from_openai(query)
    response = index.query(
        vector=query_embedding_vector,
        top_k=1,
        include_metadata=True
    )
    return response['matches'][0]['metadata']['title']
```

We use the title of the document to retrieve the document content from the disk:

```python
def load_document_content(title):
    documents_path = 'data'
    file_path = os.path.join(documents_path, title + '.txt')
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content
```

We further implement a helper method, to combine the document and the question into a prompt for the chat completion
model:

```python
def create_prompt(question, document_content):
    return 'You are given a document and a question. Your task is to answer the question based on the document.\n\n' \
           'Document:\n\n' \
           f'{document_content}\n\n' \
           f'Question: {question}'
```

Finally, we can use the OpenAI client to ask the chat completion model to answer the question based on the document.
We set the model to *gpt-3.5-turbo-16k*.
This is not the state-of-the-art model, but it is currently cheaper than the different variants of *gpt-4* and should
be sufficient for this use case.
The *16k* version of the model allows for up to 16,385 tokens, which allows us to put long texts into the prompt.
We pass a list of messages to the chat completion model, which consists of the conversation up to this point.
As we start a new conversation, our list consists of a single user message with our prompt as content.
The model returns a list of completion choices, which could be more than one if specified in the request,
but we did not specify a value, hence it defaults to only a single completion.
We extract the message content of the completion, which contains the answer to our prompt.

```python
def get_answer_from_openai(question):
    relevant_document_title = query_pinecone_index(question)
    print(f'Relevant document title: {relevant_document_title}')
    document_content = load_document_content(relevant_document_title)
    prompt = create_prompt(question, document_content)
    print(f'Prompt:\n\n{prompt}\n\n')
    completion = openai.ChatCompletion.create(
        model='gpt-3.5-turbo-16k',
        messages=[{
            'role': 'user',
            'content': prompt
        }]
    )
    return completion.choices[0].message.content

question = input('Enter a question: ')
answer = get_answer_from_openai(question)
print(answer)
```

Now we can ask questions about information from out documents and retrieve an answer from OpenAI.
As an example, we try out the following question:

> What role does the president play in the political system of Angola?

The Pinecone index yields the vector of the document *Politics of Angola* as most similar to the embedded query.
Using this document in our prompt enables OpenAI to answer the question correctly:

> The president in the political system of Angola holds almost absolute power. They are the head of state and head of government, as well as the leader of the winning party or coalition. The president appoints and dismisses members of the government, members of various courts, the Governor and Vice-Governors of the Nacional Angolan Bank, the General-Attorney and their deputies, the Governors of the provinces, and many other key positions in the government, military, police, intelligence, and security organs. The president is also responsible for defining the policy of the country and has the power to promulgate laws and make edicts. However, the president is not directly involved in making laws.

## Conclusion