class RagParameters:
    def __init__(self, chunk_size=200, overlap_size=20, retrieval_algorithm="embedding", num_retrieved_docs=3):
        self.chunk_size = chunk_size # Approximate tokens
        self.overlap_size = overlap_size # Approximate tokens
        self.retrieval_algorithm = retrieval_algorithm # "embedding", "keyword", "hybrid" (future)
        self.num_retrieved_docs = num_retrieved_docs

    def update_parameters(self, params_dict):
        """Updates parameters from a dictionary."""
        for key, value in params_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")

    @classmethod
    def from_dict(cls, params_dict):
        """Creates a RagParameters object from a dictionary."""
        params = cls()
        params.update_parameters(params_dict)
        return params