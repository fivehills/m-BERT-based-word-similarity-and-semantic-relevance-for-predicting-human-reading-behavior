
################################################################################
attention-aware measure of semantic relevance is computed within one sentence
Here take English as an example;
the word2vec pretrained database will be replaced when computing for other languages
##################################################################################


###method 1: use cosine to compute

from gensim.models import KeyedVectors
from gensim.models import Word2Vec

##other ways to load
import gensim

# Load the word vectors from the file
filename = "path/to/word/vectors/file.vec"

word_vectors = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=False)

# Get the vector for a specific word
vector = word_vectors["word"]
# obtain word list
word_list = word_vectors.index_to_key
# Define the input sentence
sentence = "The quick brown fox jumps over the lazy dog"

# Tokenize the input sentence into individual words
words = sentence.lower().split()

# Compute the similarity between two words of interest
word1 = "quick"
word2 = "lazy"


if word1 in word_list and word2 in word_list:
    similarity = word_vectors.similarity(word1, word2)
    print(f"Similarity between '{word1}' and '{word2}': {similarity:.2f}")
else:
    print(f"One or more words not in vocabulary.")


############################################################
####within one sentence
###use correlation to replace cosine similarity
####################################################################

from gensim.models import KeyedVectors
import numpy as np
import gensim

# Load the word vectors from the file
filename = "path/to/word/vectors/file.vec"

word_vectors = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=False)

# Get the vector for a specific word
#vector = word_vectors["word"]
# obtain word list
word_list = word_vectors.index_to_key
model=word_vectors
sentence = "The quick brown fox jumps over the lazy dog"
words = sentence.lower().split()

val=[]
for i in range(1, (len(words)-1)):
    if len(words)> 3:
       w1 = words[i-3]
       w2 = words[i-2]
       w3 = words[i-1]
       wt = words[i]
       wn = words[i+1]

       if w1 in word_list and w2 in word_list and wt in word_list and wn in word_list:
          # Get the word vectors
          vector1 = model[w1]
          vector2 = model[w2]
          vector3 = model[w3]
          vectort = model[wt]
          vectorn = model[wn]
          # Compute the correlation coefficient between the two vectors
          cor1 = np.corrcoef(vector1, vector2)[0, 1]
         # cor2 = np.corrcoef(vector1, vector3)[0, 1]
          cor2 = np.corrcoef(vector2, vector3)[0, 1]
          cor3 = np.corrcoef(vector3, vectort)[0, 1]
          cor4 = np.corrcoef(vectort, vectorn)[0, 1]
          cor_sum=cor1*1/3+cor2*1/2+cor3+cor4*1/2
          val.append(cor_sum)
         # print(f"Correlation between '{word1}' and '{word2}': {correlation:.2f}")
       else:
          print(f"One or more words not in vocabulary.")
val
