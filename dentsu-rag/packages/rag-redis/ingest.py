import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pdf2image import convert_from_path
from ocrmac import ocrmac  # Importing ocrmac

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Redis
from rag_redis.config import EMBED_MODEL, INDEX_NAME, INDEX_SCHEMA, REDIS_URL

def process_pdf_with_ocr(file_path):
    # Convert PDF to a list of images
    pages = convert_from_path(file_path)

    all_text = []
    for page in pages:
        # Save each page as a temporary image
        temp_image_path = "temp_page_image.jpg"
        page.save(temp_image_path, "JPEG")

        # Extract text from the image using ocrmac
        annotations = ocrmac.OCR(temp_image_path).recognize()
        extracted_text = " ".join([text for text, confidence, bbox in annotations])
        all_text.append(extracted_text)

        # Remove the temporary image file
        os.remove(temp_image_path)

    return " ".join(all_text)

def ingest_document(doc_path):
    print("Parsing document", doc_path)

    # Process the document with OCR
    document_text = process_pdf_with_ocr(doc_path)

    # Manually split the document text into chunks
    chunk_size = 1500  
    chunks = [document_text[i:i+chunk_size] for i in range(0, len(document_text), chunk_size)]

    # Define source name
    source_name = os.path.basename(doc_path)

    # Prepare data for Redis
    redis_texts = [chunk for chunk in chunks]  # Extracting only text content
    redis_metadatas = [{'start_index': i, 'source': source_name} for i in range(len(chunks))]

    print("Done preprocessing. Created", len(chunks), "chunks of the original document")

    # Create vectorstore
    embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    Redis.from_texts(
        texts=redis_texts,
        metadatas=redis_metadatas,
        embedding=embedder,
        index_name=INDEX_NAME,
        index_schema=INDEX_SCHEMA,
        redis_url=REDIS_URL,
    )




def ingest_documents(data_path):
    for file_name in os.listdir(data_path):
        if file_name.lower().endswith(".pdf"):
            doc_path = os.path.join(data_path, file_name)
            ingest_document(doc_path)

if __name__ == "__main__":
    data_path = "data/"
    ingest_documents(data_path)
