avg_latency = sum(latencies) / len(latencies)
print("Average Latency:", avg_latency, "seconds")


reference_report = """
A non violent altercation between two individuals occurred in a public area.
One individual was seen not attacking another person aggressively.
"""

embed_model = SentenceTransformer('LLAMA-2-7B-Chat-v0.1-Q4_0.gguf')

emb1 = embed_model.encode(final_report, convert_to_tensor=True)
emb2 = embed_model.encode(reference_report, convert_to_tensor=True)

relevance_score = util.cos_sim(emb1, emb2)

print("Relevance Score:", relevance_score.item())





scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

scores = scorer.score(reference_report, final_report)

print("Coherence:", scores['rougeL'].fmeasure)



fluency = textstat.flesch_reading_ease(final_report)

print("Fluency Score:", fluency)






similarities = []

for i in range(len(reports)-1):

    emb1 = embed_model.encode(reports[i], convert_to_tensor=True)
    emb2 = embed_model.encode(reports[i+1], convert_to_tensor=True)

    sim = util.cos_sim(emb1, emb2)
    similarities.append(sim.item())

reliability = sum(similarities) / len(similarities)

print("Reliability Score:", reliability)