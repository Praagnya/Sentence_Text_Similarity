# Sentence_Text_Similarity
The project involves quantifying the degree of similarity between the two corpora on Semantic similarity.
Built a model using Sentence-BERT(SBERT) to assess the degree to which two sentences are semantically equivalent to each other.

Approach to model building:
- Feed the sentence into the transformer network like BERT.
- BERT produces contextual word embeddings for all the input tokens in our text i.e. 501 word tokens for example 1(text1[0])
- As we want a fixed sized output representation, we need a pooling layer. Different pooling layers are available and the most basic one is Mean-Pooling.
- It simply averages all the contextualised word embeddings of BERT. This gives 768 dimensional output vector which is independent of how long our input vector(corpus).
- We can then use, Cosine Similarity for analysis where cosine similarity is cosine of the angle between two vectors.

Sample Model of Sentence-BERT https://www.sbert.net/docs/usage/semantic_textual_similarity.html


References: 
https://arxiv.org/pdf/1908.10084.pdf
https://www.sbert.net/docs/usage/semantic_textual_similarity.html
