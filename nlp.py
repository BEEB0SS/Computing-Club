from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

sentences = ["I love AI club.", "Machine learning is fascinating!", "Deep learning is a subset of machine learning."]
tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences)

print(padded)
