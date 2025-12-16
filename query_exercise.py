from sentence_transformers import SentenceTransformer, util

#loading the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

#10 sample texts
texts = [
    "Sri Lanka is an island nation",
    "RAG improves LLM accuracy by adding external knowledge.",
    "Python is widely used for machine learning",
    "Australia has beautiful beaches.",
    "Neural networks learn from data using layers.",
    "Cloud Computing in South Asia.",
    "Vector databases store embeddings for fast search.",
    "Security in the Metaverse is a major challenge."
    "AI helps automate business workflows."
]

#Embeding all the texts
embs = model.encode(texts, convert_to_tensor=True)

#Query

query = "How do RAG systems enhance AI?"
query_emb = model.encode(query, convert_to_tensor=True)

#compute cosine similarity
scores = util.cos_sim(query_emb, embs)[0]

#Printing the top 3 results
top3 = scores.topk(3)

print("Query:", query)
print("\nTop-3 similar texts:\n")

for score, idx in zip(top3.values, top3.indices):
    print(f"{float(score):.4f} -> {texts[idx]}")