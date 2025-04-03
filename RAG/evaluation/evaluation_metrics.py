# evaluation/evaluation_metrics.py

def calculate_source_hit(relevant_page, retrieved_chunks_text, dataset_entry):
    """
    Calculates if the source page/identifier is present in the retrieved chunks.
    Checks if the document identifier (e.g., page number) associated with the answer
    is mentioned in any of the retrieved text chunks.

    Args:
        relevant_page (str): The identifier (e.g., page number) of the source document/context
                             expected to contain the answer.
        retrieved_chunks_text (List[str]): A list of text content from the retrieved chunks.
        dataset_entry (Dict): The original dataset entry for context (currently unused here but kept for signature).

    Returns:
        bool: True if the relevant_page identifier is found in any retrieved chunk, False otherwise.
    """
    # Ensure relevant_page is treated as a string for consistent searching
    target_page_str = str(relevant_page)
    is_source_hit = False
    for chunk_text in retrieved_chunks_text:
        # Simple substring check. Consider more robust methods if needed (e.g., regex for whole words).
        if target_page_str in chunk_text:
            is_source_hit = True
            break # Found it in one chunk, no need to check others
    return is_source_hit

def evaluate_rag_pipeline(dataset, retriever, collection, rag_params):
    """
    Evaluates the RAG pipeline's retrieval component on a given dataset.
    Focuses on Source Hit Rate and tracks total questions.

    Args:
        dataset (List[Dict]): The dataset containing questions and expected source identifiers (e.g., 'page').
        retriever: An initialized retriever object (e.g., EmbeddingRetriever).
        collection: The ChromaDB collection object.
        rag_params (Dict): Dictionary containing RAG parameters like 'num_retrieved_docs'.

    Returns:
        Dict: A dictionary containing the 'source_hit_rate' (float percentage) and 'total_questions'.
    """
    total_questions = 0
    source_hits = 0 # Renamed from correct_retrievals

    for entry in dataset:
        question = entry.get('question')
        # Use .get() for safer access to 'page', default to None if missing
        relevant_page = entry.get('page')

        if not question or relevant_page is None:
            print(f"Warning: Skipping entry due to missing 'question' or 'page': {entry}")
            continue # Skip entries missing essential info

        total_questions += 1
        try:
            question_embedding = retriever.vectorize_text(question)

            # Fetch embeddings and documents - consider optimizing if performance is an issue
            results = collection.get(include=['embeddings', 'documents'])
            document_chunk_embeddings_from_db = results.get('embeddings')
            document_chunks_text_from_db = results.get('documents')

            if document_chunk_embeddings_from_db is None or document_chunks_text_from_db is None:
                print(f"Warning: Could not retrieve embeddings or documents from DB for question: {question}")
                # Decide how to handle this: count as miss, or skip? Currently counts as miss.
                is_hit = False
            else:
                retrieved_chunks_text, _ = retriever.retrieve_relevant_chunks(
                    question_embedding,
                    document_chunk_embeddings_from_db,
                    document_chunks_text_from_db,
                    top_k=rag_params.get("num_retrieved_docs", 3)
                )
                # Call the renamed function
                is_hit = calculate_source_hit(relevant_page, retrieved_chunks_text, entry)

            if is_hit:
                source_hits += 1

        except Exception as e:
            print(f"Error processing question '{question}': {e}")
            # Decide how to handle errors: count as miss, or skip? Currently counts as miss.

    # Calculate rate, handle division by zero
    source_hit_rate = (source_hits / total_questions) * 100 if total_questions > 0 else 0.0

    # Return results with the new key name
    return {"source_hit_rate": source_hit_rate, "total_questions": total_questions}