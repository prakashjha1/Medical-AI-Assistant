from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document

def load_pdf_files(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )

    documents = loader.load()
    return documents

def extract_minimal_documents(documents: List[Document]) -> List[Document]:
    """
    Accepts a list of Document objects and returns a new list containing
    only the original content and its 'source' metadata.
    """
    result: List[Document] = []
    for doc in documents:
        source = doc.metadata.get("source")
        simplified = Document(
            page_content=doc.page_content,
            metadata={"source": source}
        )
        result.append(simplified)
    return result

def split_text_for_contextual_integrity(docs, chunk_size=500, chunk_overlap=50):
    """
    Split documents into chunks that preserve natural and semantic boundaries.

    Parameters:
        docs (List[Document]): Input documents to chunk.
        chunk_size (int): Approximate maximum characters per chunk.
        chunk_overlap (int): Desired overlap between adjacent chunks.

    Returns:
        List[Document]: Chunked documents maintaining context and coherence.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],  # Prefer paragraphs, then sentences
        length_function=len
    )
    return splitter.split_documents(docs)

def download_embeddings():
    """
    Download and return the HuggingFace embeddings model.
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name
    )
    return embeddings