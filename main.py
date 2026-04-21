from Ingestion import DataObject, Ingest, MimeType
from Chunking import Chunk, Chunker
from pprint import pprint
from dotenv import load_dotenv

if __name__ == "__main__":

    load_dotenv()

    file_path = "commentary_excerpts.txt"
    mime = MimeType.TXT
    ingester = Ingest(file_path,mime)
    data = ingester.execute()

    chunker = Chunker(data)
    chunks = chunker._txt_splitter()

    for chunk in chunks:
        pprint(chunk.__dict__)
        print(chunk.embedding)
        break
