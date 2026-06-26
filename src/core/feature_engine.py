import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, models
from sklearn.metrics.pairwise import cosine_similarity



class SimpleEmbeddingEngine:
    """
    A minimal, self-contained semantic search and metadata manager that enables 
    the following requirements:
    - Indexes a corpus of sentences 
    - Supports semantic similarity search (e.g., "find the most similar sentences") 
    - Optionally includes a classification component to detect "unfair clauses"
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """
        Initializes the dense semantic vector encoder framework.

        1. Spins up a baseline SentenceTransformer token mapping engine.
        2. Sets up standard dimensional spaces for general-purpose inference.
        3. Prepares internal placeholders for indexing textual terms of service.

        Args:
            model_name (str, optional): The Hugging Face identifier for the 
                pretrained transformer sequence to download and load into memory. 
                Defaults to "all-MiniLM-L6-v2".

        Returns:
            None: Allocates hardware and model states directly onto the engine context.
        """        
        logging.info(f"Loading SentenceTransformer vector mapping space: {model_name}")
        self.encoder = SentenceTransformer(model_name)
        self.corpus_embeddings = None
        self.metadata_df = None


    def transform(self, texts: list[str] | str) -> np.ndarray:
        """
        Encodes raw text or a list of text strings into the precise float32 matrix 
        structure expected by downstream classifiers and matrix similarity steps.

        1. Validates and ensures the text encoder framework is correctly warmed up.
        2. Normalizes varying scalar shapes by coercing raw strings into a list format array.
        3. Projects the sequence data through the transformer blocks to map semantic traits.

        Args:
            texts (list[str] | str): A single raw sentence string or a collection list 
                of text sentences to be processed through the semantic feature pipeline.

        Returns:
            np.ndarray: A precise 2D continuous feature space matrix of float32 
                embeddings ready for downstream classification or distance comparisons.
        """
        if self.encoder is None:
            raise ValueError("Encoder framework is uninitialized.")
        
        if isinstance(texts, str):
            texts = [texts]
            
        return self.encoder.encode(texts, convert_to_numpy=True).astype("float32")

    def build_index(self, df: pd.DataFrame) -> np.ndarray:
        """
        Encodes a textual dataframe corpus into a fixed continuous feature space matrix.

        1. Extracts structural data attributes and routes text blocks through the encoder pipeline.
        2. Caches calculated dense tensor sequences directly into active memory arrays.
        3. Creates isolated lookup maps for historic attributes including company origins and labels.

        Args:
            df (pd.DataFrame): A collection source frame containing columns for 'text', 
                'company', 'sentence_idx', and binary 'is_unfair' flags.

        Returns:
            np.ndarray: The finalized numerical matrix block representing the indexed 
                semantic coordinates of the full text corpus.
        """
        logging.info(f"Transforming {len(df)} lines into a dense feature space...")
        
        self.corpus_embeddings = self.transform(df['text'].tolist())
        
        self.metadata_df = df[['company', 'sentence_idx', 'text', 'is_unfair']].copy().reset_index(drop=True)
        return self.corpus_embeddings

    def search(self, query_text: str, top_k: int = 3) -> list[dict]:
        """
        Performs a spatial proximity query using basic cosine similarity scoring.

        1. Converts user target text patterns into matching continuous dense feature arrays.
        2. Evaluates distance metrics across the precomputed historical matrix arrays.
        3. Ranks historical matches based on proximity weights and combines original records.

        Args:
            query_text (str): The raw evaluation clause sentence to look up across the database.
            top_k (int, optional): Total maximum count of close matching documents to return. 
                Defaults to 3.

        Returns:
            list[dict]: A ranked array of data structures containing close semantic texts, 
                original ground-truth parameters, and calculated metric scores.
        """
        if self.corpus_embeddings is None or self.metadata_df is None:
            raise ValueError("Engine states must be initialized or loaded before running queries.")
            
        query_vector = self.transform(query_text)
        
        scores = cosine_similarity(query_vector, self.corpus_embeddings)[0]
        top_indices = scores.argsort()[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            row = self.metadata_df.iloc[idx].to_dict()
            row['similarity_score'] = float(scores[idx])
            results.append(row)
            
        return results
            
    def save_artifacts(self, output_dir: Path) -> None:
        """
        Serializes the active numerical matrix and metadata lookup tables directly to disk.

        1. Ensures baseline indexes are actively instantiated in working system buffers.
        2. Flushes the dense vector feature spaces into continuous binary numpy files.
        3. Pickles structured lookups to preserve structural schemas outside active runtimes.

        Args:
            output_dir (Path): Explicit directory filesystem coordinates indicating where 
                the compiled analytical states should be deposited.

        Returns:
            None: Commits cached structures onto stable storage layers.
        """
        if self.corpus_embeddings is None or self.metadata_df is None:
            raise ValueError("No calculated states found to save. Run build_index first.")
            
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Save the numerical matrix efficiently as binary numpy format
        np.save(output_dir / "embeddings.npy", self.corpus_embeddings)
        
        # 2. Save the metadata dataframe structure as a pickle file
        self.metadata_df.to_pickle(output_dir / "metadata.pkl")
        
        logging.info(f"Successfully saved all search artifacts to {output_dir}")

    def load_artifacts(self, output_dir: Path) -> None:
        """
        Loads pre-computed matrix configurations and dataframe records into RAM memory blocks.

        1. Inspects designated storage nodes to confirm tracking targets exist.
        2. Pulls saved binary float32 tensor records back into system layout schemas.
        3. Restores relational tracking variables to make engines instantly responsive.

        Args:
            output_dir (Path): The source storage folder path from which serialized 
                binary assets and pickles will be read.

        Returns:
            None: Instantiates model state memories across active runtime frames.
        """        
        emb_path = output_dir / "embeddings.npy"
        meta_path = output_dir / "metadata.pkl"
        
        if not emb_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"Required artifacts missing in target directory: {output_dir}")
            
        self.corpus_embeddings = np.load(emb_path)
        self.metadata_df = pd.read_pickle(meta_path)
        
        logging.info(f"Engine warmed up successfully from disk! Loaded {len(self.metadata_df)} vectors.")






class LegalBertEmbeddingEngine:
    """
    A minimal, self-contained semantic search and metadata manager that replaces
    the general-purpose framework with a domain-specific LEGAL-BERT pipeline.
    
    Maintains exact API alignment (inputs/outputs) with the prior engine structure.
    """
    
    def __init__(self, model_name: str = "nlpaueb/legal-bert-base-uncased") -> None:
        """
        Initializes the specialized dense Legal-BERT vector encoder space

        1. Suppresses verbose Hugging Face initialization streams to ensure clean runtime output.
        2. Spawns the underlying contextual token transformer block using target model weights.
        3. Chains an explicit mean-pooling layer to contract sequence lengths into stable structures.
        4. Compiles the modular sentence-level encoder execution pipeline.

        Args:
            model_name (str): The Hugging Face registry key or local directory route pointing 
                to the specialized target base transformer. Defaults to "nlpaueb/legal-bert-base-uncased".

        Raises:
            Exception: Encapsulates and passes up any file system, memory allocation, or network 
                connectivity failures encountered during the weight assembly process.
        """

        logging.info(f"Assembling Legal-BERT vector mapping space using base model: {model_name}")
        
        try:
            # Temporarily quiet the Hugging Face transformers logging output stream
            import transformers
            hf_logger_level = transformers.utils.logging.get_verbosity()
            transformers.utils.logging.set_verbosity_error()

            # 1. Load the core transformer, explicitly telling it not to leak loading info structures
            word_embedding_model = models.Transformer(
                model_name, 
                model_args={"output_loading_info": False}
            )
            
            # Restore standard logging verbs for the remainder of the runtime session
            transformers.utils.logging.set_verbosity(hf_logger_level)
            
            # 2. Add an explicit Mean Pooling layer to convert word pieces into single sentence vectors
            pooling_model = models.Pooling(
                word_embedding_model.get_embedding_dimension(),
                pooling_mode='mean',
            )
            
            # 3. Assemble them into an encoder that behaves identically to your original setup
            self.encoder = SentenceTransformer(modules=[word_embedding_model, pooling_model])
            
        except Exception as e:
            logging.error(f"Failed to instantiate Legal-BERT sequence blocks: {e}")
            raise e

        self.corpus_embeddings = None
        self.metadata_df = None


    def transform(self, texts: list[str] | str) -> np.ndarray:
        """
        Encodes raw text or a list of text strings into the precise float32 matrix 
        structure expected by downstream classifiers and matrix similarity steps.

        1. Validates and ensures the text encoder framework is correctly warmed up.
        2. Normalizes varying scalar shapes by coercing raw strings into a list format array.
        3. Projects the sequence data through the transformer blocks to map semantic traits.

        Args:
            texts (list[str] | str): A single raw sentence string or a collection list 
                of text sentences to be processed through the semantic feature pipeline.

        Returns:
            np.ndarray: A precise 2D continuous feature space matrix of float32 
                embeddings ready for downstream classification or distance comparisons.
        """
        if self.encoder is None:
            raise ValueError("Encoder framework is uninitialized.")
        
        # Coerce a single string input into a list format to guarantee a 2D matrix shape output
        if isinstance(texts, str):
            texts = [texts]
            
        # The output matrix dimension will shift from 384 (MiniLM) to 768 (LEGAL-BERT-BASE)
        return self.encoder.encode(texts, convert_to_numpy=True).astype("float32")


    def build_index(self, df: pd.DataFrame) -> np.ndarray:
        """Encodes a textual dataframe corpus into a fixed continuous feature space matrix."""
        logging.info(f"Transforming {len(df)} lines into a dense feature space...")
        
        # CALLS THE NEW TRANSFORM METHOD INTERNALLY 
        self.corpus_embeddings = self.transform(df['text'].tolist())
        
        self.metadata_df = df[['company', 'sentence_idx', 'text', 'is_unfair']].copy().reset_index(drop=True)
        return self.corpus_embeddings


    def search(self, query_text: str, top_k: int = 3) -> list[dict]:
        """
        Performs a spatial proximity query using basic cosine similarity scoring.

        1. Converts user target text patterns into matching continuous dense feature arrays.
        2. Evaluates distance metrics across the precomputed historical matrix arrays.
        3. Ranks historical matches based on proximity weights and combines original records.

        Args:
            query_text (str): The raw evaluation clause sentence to look up across the database.
            top_k (int, optional): Total maximum count of close matching documents to return. 
                Defaults to 3.

        Returns:
            list[dict]: A ranked array of data structures containing close semantic texts, 
                original ground-truth parameters, and calculated metric scores.
        """
        if self.corpus_embeddings is None or self.metadata_df is None:
            raise ValueError("Engine states must be initialized or loaded before running queries.")
            
        # ALSO CALLS THE UNIFIED TRANSFORM METHOD
        query_vector = self.transform(query_text)
        
        scores = cosine_similarity(query_vector, self.corpus_embeddings)[0]
        top_indices = scores.argsort()[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            row = self.metadata_df.iloc[idx].to_dict()
            row['similarity_score'] = float(scores[idx])
            results.append(row)
            
        return results
        

    def save_artifacts(self, output_dir: Path) -> None:
        """
        Serializes the active numerical matrix and metadata lookup tables directly to disk.

        1. Ensures baseline indexes are actively instantiated in working system buffers.
        2. Flushes the dense vector feature spaces into continuous binary numpy files.
        3. Pickles structured lookups to preserve structural schemas outside active runtimes.

        Args:
            output_dir (Path): Explicit directory filesystem coordinates indicating where 
                the compiled analytical states should be deposited.

        Returns:
            None: Commits cached structures onto stable storage layers.
        """
        if self.corpus_embeddings is None or self.metadata_df is None:
            raise ValueError("No calculated states found to save. Run build_index first.")
            
        output_dir.mkdir(parents=True, exist_ok=True)
        
        np.save(output_dir / "embeddings.npy", self.corpus_embeddings)
        
        self.metadata_df.to_pickle(output_dir / "metadata.pkl")
        
        logging.info(f"Successfully saved all search artifacts to {output_dir}")


    def load_artifacts(self, output_dir: Path) -> None:
        """
        Loads pre-computed matrix configurations and dataframe records into RAM memory blocks.

        1. Inspects designated storage nodes to confirm tracking targets exist.
        2. Pulls saved binary float32 tensor records back into system layout schemas.
        3. Restores relational tracking variables to make engines instantly responsive.

        Args:
            output_dir (Path): The source storage folder path from which serialized 
                binary assets and pickles will be read.

        Returns:
            None: Instantiates model state memories across active runtime frames.
        """        
        emb_path = output_dir / "embeddings.npy"
        meta_path = output_dir / "metadata.pkl"
        
        if not emb_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"Required artifacts missing in target directory: {output_dir}")
            
        self.corpus_embeddings = np.load(emb_path)
        self.metadata_df = pd.read_pickle(meta_path)
        
        logging.info(f"Engine warmed up successfully from disk! Loaded {len(self.metadata_df)} vectors.")