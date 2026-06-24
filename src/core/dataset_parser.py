import logging
import zipfile
import pandas as pd
from pathlib import Path


CATEGORIES_DICT = {
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



def extract_all_lines_from_txt_in_zipfile(archive: zipfile.ZipFile, internal_path: str) -> list[str]:
    """ For a given txt file inside the zip, extracts a list of all lines (each line is list item)

    Args:
        archive (zipfile.ZipFile): target zipfile objetct
        internal_path (str): internal path to target txt file inside each file

    Returns:
        list[str]: list of all lines in target txt file
    """
    try:
        with archive.open(internal_path) as f:
            return [line.decode("utf-8", errors="ignore").strip() for line in f.readlines()]
    except KeyError:
        return []


def parse_claudette_zipfile(zip_filepath: str) -> pd.DataFrame:
    """ Given the ToS Claudette zip file, extracts all sentences for all companies in the dataset and corresponding 
    labels into a dataframe

    (SEE CATEGORIES_DICT for labels mapping)

    Args:
        zip_filepath (str): path to target ToS Claudette ToS zipfile

    Returns:
        pd.DataFrame: dataframe containing all sentences for all companies and corresponding labels
    """
    all_sentences_df=pd.DataFrame({})

    with zipfile.ZipFile(zip_filepath, "r") as archive:
        
        logging.info('Getting all files in zip...')
        all_files_in_zip = archive.namelist()
        
        
        logging.info('Getting all sentence files in zip...')
        sentence_files = [
            f for f in all_files_in_zip
            if "/sentences/" in f.lower() and f.endswith(".txt")
        ]


        logging.info('looping all sentence files from all companies...')
        for sentence_filepath in sentence_files:

            sentence_dict={}

            filename = Path(sentence_filepath).name
            company = Path(filename).stem
            logging.info(f"Processing company {company}...")

            logging.info("Extracting all sentences...")
            sentences=extract_all_lines_from_txt_in_zipfile(archive,sentence_filepath)

            sentence_dict['company']=[company]* len(sentences)
            sentence_dict['sentence_idx'] = list(range(len(sentences)))
            sentence_dict['text']=sentences        
            for col_name, folder in CATEGORIES_DICT.items():        
                logging.info(f"Extracting all corresponding labels for {col_name}...")
                labels_filepath=f"{sentence_filepath.split('/')[0]}/{folder}/{filename}"
                raw_lines=extract_all_lines_from_txt_in_zipfile(archive,labels_filepath)
                sentence_dict[col_name] = [int(line.strip()) if line.strip().replace('-', '').isdigit() and line.strip() != '-1' else 0 for line in raw_lines]                
            
            company_df=pd.DataFrame(sentence_dict)
            all_sentences_df=pd.concat([all_sentences_df,company_df],ignore_index=True)


    return all_sentences_df