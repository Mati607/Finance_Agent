from Ingestion import DataObject, Ingest, MimeType
from Chunking import Chunk, Chunker
from Retrieval import SemanticContext, NumericContext, Retriever
from pprint import pprint
from dotenv import load_dotenv
from Agent import Agent

if __name__ == "__main__":

    load_dotenv()

    file_paths = ["commentary_excerpts.txt","bu_financials_q3.csv"]
    mimes = [MimeType.TXT,MimeType.CSV]

    retriever = Retriever(file_paths,mimes,3,Ingest, Chunker)
    agent = Agent(retriever)

    query = "Give me the financial data where the ramp was less than expected for LATAM"
    ans = agent.generate_response(query)

    print(ans)
    

