import os
from dotenv import load_dotenv
from pinecone import (
    Pinecone,
    ServerlessSpec,
    CloudProvider,
    AwsRegion,
    Metric,
    VectorType,
    DeletionProtection
)
from langchain_pinecone import PineconeVectorStore

from src.helper import (
    load_pdf_files,
    extract_minimal_documents,
    split_text_for_contextual_integrity,
    download_embeddings
)

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

extracted_data=load_pdf_files(data='data/')
filter_data = extract_minimal_documents(extracted_data)
text_chunks=split_text_for_contextual_integrity(filter_data)

embeddings = download_embeddings()


pinecone_api_key = os.environ.get("PINECONE_API_KEY")
if not pinecone_api_key:
    raise ValueError("Environment variable 'PINECONE_API_KEY' is not set")

pc = Pinecone(api_key=pinecone_api_key)

index_name = "medical-assistant"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric=Metric.COSINE,
        vector_type=VectorType.DENSE,
        spec=ServerlessSpec(
            cloud=CloudProvider.AWS,
            region=AwsRegion.US_EAST_1
        ),
        deletion_protection=DeletionProtection.DISABLED,
        tags={
            "purpose": "medical-assistant"
        }
    )
index = pc.Index(index_name)

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings, 
)