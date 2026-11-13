# Generative AI, LLMs and RAG Notes

## Retrieval-Augmented Generation

### Q: [easy] What is Retrieval-Augmented Generation (RAG) and why use it?
RAG combines a retriever with a language model. At question time the system searches an external knowledge base for passages relevant to the query, adds them to the prompt as context, and the model generates an answer grounded in that material. It reduces hallucination, lets the model use private or up-to-date information without retraining, allows answers to cite their sources, and is usually cheaper and easier to update than fine-tuning because you only re-index documents when knowledge changes.

### Q: [medium] Describe the main components of a RAG pipeline.
A RAG pipeline has an ingestion stage and a query stage. Ingestion loads documents, splits them into chunks, converts each chunk into an embedding vector, and stores the vectors with metadata in a vector index. At query time the question is embedded, the retriever finds the top-k most similar chunks, optionally a re-ranker reorders them, the chunks are assembled into a prompt, and the language model generates an answer with citations. Evaluation and monitoring wrap around both stages.

### Q: [medium] How do you choose chunk size and overlap?
Chunks must be small enough to be specific and fit several into the prompt, but large enough to carry complete ideas. Very small chunks lose context; very large chunks dilute the embedding and waste tokens. Typical sizes are a few hundred tokens with about 10 to 20 percent overlap so that sentences cut at a boundary still appear whole in a neighbouring chunk. Splitting on natural structure such as headings and paragraphs usually beats fixed-size splits. Tune the values by measuring retrieval quality on a set of test questions.

### Q: [medium] What is hybrid search and why does it help?
Hybrid search combines dense retrieval, which uses embeddings to match meaning, with sparse retrieval such as BM25, which matches exact keywords. Dense search handles paraphrases but can miss rare terms, acronyms and identifiers; keyword search catches those exact matches but misses synonyms. The two ranked lists are merged, commonly with Reciprocal Rank Fusion, which scores each document as the sum of one divided by a constant plus its rank in each list. Hybrid retrieval is usually more robust than either method alone.

### Q: [hard] How would you evaluate and improve a RAG system?
Evaluate retrieval and generation separately. For retrieval, build a labelled set of questions with known relevant passages and measure recall at k, precision, mean reciprocal rank and nDCG. For generation, measure faithfulness, meaning the answer is supported by the retrieved context, answer relevance, and correctness against reference answers, using human review or an LLM judge. To improve it: tune chunking, add hybrid search and a cross-encoder re-ranker, rewrite or expand queries, add metadata filters, use better embedding models, and instruct the model to say when the context does not contain the answer.

## LLM concepts

### Q: [medium] What are embeddings and how is similarity measured?
An embedding is a dense vector that represents text so that items with similar meaning are close together in the vector space. Similarity is usually measured with cosine similarity, the cosine of the angle between two vectors, which ignores vector length; when vectors are normalised it equals the dot product. Other measures are Euclidean distance and inner product. Embeddings power semantic search, clustering, recommendation and RAG retrieval.

### Q: [medium] How do you reduce hallucinations in LLM applications?
Ground the model in retrieved or supplied source material and instruct it to answer only from that context. Require citations, lower the temperature, and allow the model to answer that it does not know. Add verification steps such as checking claims against sources or using a second model as a judge, constrain outputs with structured formats, and keep prompts short and specific. For high-stakes use, add human review and monitoring.

### Q: [hard] When would you fine-tune a model instead of using RAG?
Use RAG when the problem is missing or changing knowledge, when you need citations, or when data is private and frequently updated. Fine-tune when you need to change behaviour rather than facts: a consistent style or format, a specialised task, domain-specific terminology, or lower latency and cost by distilling into a smaller model. Fine-tuning bakes knowledge into weights, so it is harder to update and cannot cite sources. The two are complementary, and many production systems combine a fine-tuned model with retrieval.
