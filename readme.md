Hybrid Search Langchain
!pip install --upgrade --quiet  pinecone-client pinecone-text pinecone-notebooks
api_key="pcsk_ctUXr_6C1LD2Q4b8AjvYqmhfDABUjgJpFVNaqAcTnyaAT87UYK4geYq829f4At9yhynfw"
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone import Pinecone, ServerlessSpec
import os

api_key = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=api_key)  #  Initialize client

index_name = "hybrid-search-langchain-pinecone"

if not pc.has_index(name=index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="dotproduct",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print(f"Created index {index_name}")
else:
    print(f"Index {index_name} already exists")
Created index hybrid-search-langchain-pinecone
index=pc.Index(index_name)
index
<pinecone.db_data.index.Index at 0x205ab4f3710>
## vector embedding and sparse matrix

import os
from dotenv import load_dotenv
load_dotenv()

os.environ["HF_TOKEN"]=os.getenv("HF_TOKEN")


from langchain_huggingface import HuggingFaceEmbeddings

embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
embeddings
WARNING:tensorflow:From C:\Users\aryan\AppData\Roaming\Python\Python312\site-packages\tf_keras\src\losses.py:2976: The name tf.losses.sparse_softmax_cross_entropy is deprecated. Please use tf.compat.v1.losses.sparse_softmax_cross_entropy instead.

HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2', cache_folder=None, model_kwargs={}, encode_kwargs={}, query_encode_kwargs={}, multi_process=False, show_progress=False)
from pinecone_text.sparse import BM25Encoder

bm25_encoder=BM25Encoder().default()
bm25_encoder
<pinecone_text.sparse.bm25_encoder.BM25Encoder at 0x20604fdcaa0>
sentences=[
    "In 2023, I visited Paris",
        "In 2022, I visited New York",
        "In 2021, I visited New Orleans",

]

## tfidf values on these sentence

bm25_encoder.fit(sentences)

## store the values to a json file

bm25_encoder.dump("bm25_values.json")

# load to your BM25Encoder object

bm25_encoder = BM25Encoder().load("bm25_values.json")
  0%|          | 0/3 [00:00<?, ?it/s]
retriever=PineconeHybridSearchRetriever(embeddings=embeddings,sparse_encoder=bm25_encoder,index=index)
retriever
PineconeHybridSearchRetriever(embeddings=HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2', cache_folder=None, model_kwargs={}, encode_kwargs={}, query_encode_kwargs={}, multi_process=False, show_progress=False), sparse_encoder=<pinecone_text.sparse.bm25_encoder.BM25Encoder object at 0x000002060635FDD0>, index=<pinecone.db_data.index.Index object at 0x00000205AB4F3710>)
retriever.add_texts(
    [
    "In 2023, I visited Paris",
        "In 2022, I visited New York",
        "In 2021, I visited New Orleans",

]
)
  0%|          | 0/1 [00:00<?, ?it/s]
retriever.invoke("What city did i visit first")
[Document(metadata={'score': 0.23936905}, page_content='In 2021, I visited New Orleans'),
 Document(metadata={'score': 0.232818261}, page_content='In 2022, I visited New York'),
 Document(metadata={'score': 0.21249935}, page_content='In 2023, I visited Paris')]
 