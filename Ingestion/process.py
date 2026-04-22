import pandas as pd
from enum import Enum

class MimeType(Enum):
    CSV = 1
    TXT = 2
    XLSX = 3

class DataObject:
    def __init__(
            self, 
            data: pd.DataFrame, 
            schema: str, 
            document_type: MimeType,
            filename :str
    ):
        self.data = data
        self.schema = schema
        self.document_type = document_type
        self.filename = filename


class Ingest:
    def __init__(
        self,
        path : str, 
        document_type : MimeType
    ):
        self.path = path
        self.document_type = document_type

    def _generate_schema(self, df: pd.DataFrame) -> dict[str,list]:
        schema = {
            "columns" : [
                {
                    "name": col,
                    "dtype" : str(df[col].dtype)
                }
                for col in df.columns
            ]
        }
        return schema
    
    @property
    def _ingest_csv(self) -> DataObject:
        df = pd.read_csv(self.path)
        schema = self._generate_schema(df)
        doc_type = MimeType.CSV
        return DataObject(
            df,
            schema,
            doc_type,
            self.path
        )
        
    
    @property
    def _ingest_xlsx(self) -> DataObject:
        df = pd.read_excel(self.path)
        schema = self._generate_schema(df)
        doc_type = MimeType.XLSX
        return DataObject(
            df,
            schema,
            doc_type,
            self.path
        )
        
    
    @property
    def _ingest_txt(self) -> DataObject:
        with open(self.path, "r", encoding="utf-8") as f:
            text = f.read()
        df = pd.DataFrame({"content": pd.Series([text],dtype="string")})
        schema = self._generate_schema(df)
        doc_type = MimeType.TXT
        return DataObject(
            df,
            schema,
            doc_type,
            self.path
        )

    def execute(self) -> DataObject:
        if self.document_type == self.document_type.CSV:
            return self._ingest_csv

        if self.document_type == self.document_type.XLSX:
            return self._ingest_xlsx

        if self.document_type == self.document_type.TXT:
            return self._ingest_txt
        
