from langchain_core.documents import Document

from langchain_rag.splitter import LangChainSplitter


def test_split_documents_tags_source():
    """split_documents accepts Document objects and tags source metadata."""
    docs = [
        Document(page_content="First paragraph about Python."),
        Document(page_content="Second paragraph about RAG."),
    ]

    splitter = LangChainSplitter(chunk_size=50, chunk_overlap=5)
    result = splitter.split_documents(docs, source="my_notes.txt")

    assert all(isinstance(d, Document) for d in result)
    assert all(d.metadata["source"] == "my_notes.txt" for d in result)

def test_split_documents_no_source_preserves_existing():
    """Without source arg, existing metadata is kept."""
    docs = [
        Document(page_content="Hello world.", metadata={"source": "existing.txt"}),
    ]

    splitter = LangChainSplitter(chunk_size=200, chunk_overlap=0)
    result = splitter.split_documents(docs)

    assert result[0].metadata["source"] == "existing.txt"