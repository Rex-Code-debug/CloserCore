from bs4 import BeautifulSoup
import requests
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

def scrape_website(url: str):
    print(f"👀 Reading: {url}")
    session = requests.Session()
    # Mocking a real browser
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36"
    })
    try:
        response = session.get(url, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            # Kill script and style elements
            for script in soup(["script", "style"]):
                script.extract()    

            text = soup.get_text(separator="\n")
            # Clean empty lines
            clean_text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
            return clean_text[2000:7000] # Truncate to save context
        else:
            return f"Error: Status code {response.status_code}"
    except Exception as e:
        return f"Error: Could not scrape {url}. Reason: {str(e)}"

def retriever_text(text):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    if not text or text.startswith("Error"):
        return None, None
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = splitter.create_documents([text])
    
    return chunks, embeddings

def get_price_chunks(chunks, embeddings, query="price $ or ₹", top_k=5):
    """
    Retrieve top k chunks most relevant to pricing information
    """
    # Create vector store from chunks
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Search for chunks similar to pricing-related query
    price_chunks = vectorstore.similarity_search(query, k=top_k)
    
    return price_chunks
