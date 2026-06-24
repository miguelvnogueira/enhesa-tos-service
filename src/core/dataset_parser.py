import io
import zipfile
from pathlib import Path
import pandas as pd

class ClaudetteParser:
    """Parses the CLAUDETTE corpus directly from memory out of a nested zip file."""
    
    def __init__(self, zip_path: str | Path):
        self.zip_path = Path(zip_path)
        
        # Categorical folder names mapping to internal structures
        self.categories = {
            "is_unfair": "Labels",
            "arbitration": "Labels_A",
            "unilateral_change": "Labels_CH",
            "content_removal": "Labels_CR",
            "jurisdiction": "Labels_J",
            "choice_of_law": "Labels_LAW",
            "limitation_of_liability": "Labels_LTD",
            "unilateral_termination": "Labels_TER",
            "contract_by_using": "Labels_USE"
        }

    def _read_zip_lines(self, archive: zipfile.ZipFile, internal_path: str) -> list[str]:
        try:
            with archive.open(internal_path) as f:
                return [line.decode("utf-8", errors="ignore").strip() for line in f.readlines()]
        except KeyError:
            return []

    def parse(self) -> pd.DataFrame:
        records = []
        
        with zipfile.ZipFile(self.zip_path, "r") as archive:
            all_files = archive.namelist()
            
            # Scans for the text files anywhere inside the zip, bypassing any top-level wrapper directory
            sentence_files = [
                f for f in all_files 
                if "/sentences/" in f.lower() and f.endswith(".txt")
            ]
            
            if not sentence_files:
                # If it's a flat zip file without a parent directory wrapper
                sentence_files = [f for f in all_files if f.lower().startswith("sentences/") and f.endswith(".txt")]
                
            if not sentence_files:
                raise ValueError(f"Could not find any 'Sentences' directory inside the zip file structure.")

            for file_name in sentence_files:
                pure_filename = Path(file_name).name  # e.g., "9gag.txt"
                company = Path(file_name).stem       # e.g., "9gag"
                
                sentences = self._read_zip_lines(archive, file_name)
                
                # Match the label matrices using relative lookups to account for the parent folder
                labels_matrix = {}
                for col_name, folder in self.categories.items():
                    # Safely look up the exact internal path for the target label file
                    internal_label_path = next(
                        (f for f in all_files if f.lower().endswith(f"{folder.lower()}/{pure_filename.lower()}")),
                        None
                    )
                    
                    lines = self._read_zip_lines(archive, internal_label_path) if internal_label_path else []
                    labels_matrix[col_name] = [
                        int(val) if val.replace('-', '').isdigit() else -1 
                        for val in lines
                    ]

                # Align lines cleanly
                for idx, sentence in enumerate(sentences):
                    if not sentence: 
                        continue  
                    
                    row = {
                        "company": company,
                        "sentence_idx": idx,
                        "text": sentence
                    }
                    
                    for col_name in self.categories.keys():
                        stream = labels_matrix[col_name]
                        val = stream[idx] if idx < len(stream) else -1
                        row[col_name] = 1 if val in (2, 3) else 0
                    
                    records.append(row)
                    
        return pd.DataFrame(records)