import logging

def process_user_sentence(search_engine, clause_clf, input_sentence: str, top_k: int=3):

    logging.info(f"Processing input clause: '{input_sentence}'")

    logging.info(f"Searching for top {top_k} similarity matches...")
    search_results = search_engine.search(input_sentence, top_k=top_k)

    logging.info(f"Classifying sentence as fair or unfair...")
    query_vector = search_engine.transform(input_sentence)

    prediction = clause_clf.predict(query_vector)


    return search_results, prediction

