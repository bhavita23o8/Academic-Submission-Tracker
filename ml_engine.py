import logging
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer, util
    MODEL_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    util = None
    MODEL_AVAILABLE = False
    logger.warning("sentence-transformers not installed. ML features will be disabled.")

class MLEngine:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MLEngine, cls).__new__(cls)
            cls._instance._load_model()
        return cls._instance

    def _load_model(self):
        if MODEL_AVAILABLE and self._model is None:
            try:
                logger.info("Loading Sentence-BERT model... (This may take a moment on first run)")
                # 'all-MiniLM-L6-v2' is a great balance of speed and accuracy
                self._model = SentenceTransformer('all-MiniLM-L6-v2') 
                logger.info("Sentence-BERT model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                self._model = None

    def calculate_similarity(self, text1, text2):
        """
        Calculates semantic similarity between two texts using SBERT.
        Returns a score between 0.0 and 1.0.
        """
        if not MODEL_AVAILABLE or self._model is None:
            logger.error("ML Model not available for similarity check.")
            return 0.0

        if not text1 or not text2:
            return 0.0

        try:
            # Compute embeddings
            embeddings1 = self._model.encode(text1, convert_to_tensor=True)
            embeddings2 = self._model.encode(text2, convert_to_tensor=True)

            # Compute cosine similarity
            cosine_score = util.cos_sim(embeddings1, embeddings2)
            
            # Convert to float
            return float(cosine_score[0][0])
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0

# Singleton instance
ml_engine = MLEngine()
