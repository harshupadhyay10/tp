from sentence_transformers import SentenceTransformer
sentences = ["Adv-TopUpData-10GIG-24 Mths", "Adv-TopUpData-175MB-24 Mths", "Adv-TopUpData-1.5GIG-24 Mths", "ADSL 5-GIG Shaped 24 Months"]

model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')
embeddings = model.encode(sentences)
print(embeddings)

from sklearn.metrics.pairwise import cosine_similarity
#let's calculate cosine similarity for sentence 0:
print(cosine_similarity([embeddings[0]],embeddings[1:]))