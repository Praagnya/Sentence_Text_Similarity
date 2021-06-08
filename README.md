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
```python
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('paraphrase-MiniLM-L12-v2')

# Two lists of sentences
sentences1 = ['The cat sits outside',
             'A man is playing guitar',
             'The new movie is awesome']

sentences2 = ['The dog plays in the garden',
              'A woman watches TV',
              'The new movie is so great']

#Compute embedding for both lists
embeddings1 = model.encode(sentences1, convert_to_tensor=True)
embeddings2 = model.encode(sentences2, convert_to_tensor=True)

#Compute cosine-similarits
cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)

#Output the pairs with their score
for i in range(len(sentences1)):
    print("{} \t\t {} \t\t Score: {:.4f}".format(sentences1[i], sentences2[i], cosine_scores[i][i]))
```


References: 
https://arxiv.org/pdf/1908.10084.pdf
https://www.sbert.net/docs/usage/semantic_textual_similarity.html
