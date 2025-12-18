from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import SentenceTransformerEmbeddings

loader = PyPDFLoader("sample.pdf")
docs = loader.load()

text = """
We encourage pinning your version to a specific version in order to avoid breaking your CI when we publish new tests. We recommend upgrading to the latest version periodically to make sure you have the latest tests.

Not pinning your version will ensure you always have the latest tests, but it may also break your CI if we introduce tests that your integration doesn't pass.

💁 Contributing
As an open-source project in a rapidly developing field, we are extremely open to contributions, whether it be in the form of a new feature, improved infrastructure, or better documentation.
"""

char_text_splitter = CharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=25,
    separator="\n",
)

recursive_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
)

#result = char_text_splitter.split_documents(docs)
#print(result[0].page_content)
#result = recursive_text_splitter.split_text(text=docs[0].page_content)
#print(result[1])


# 1. Load Embeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Initialize SemanticChunker with Breakpoint Strategy
splitter = SemanticChunker(embeddings, breakpoint_threshold_type="standard_deviation",
                           breakpoint_threshold_amount=0.9)

# 3. Split Documents
result = splitter.split_documents(docs)
print(result)