import nltk
nltk.download('wordnet')

from nltk.corpus import wordnet

# 단어 찾기
cars = wordnet.synsets('car')
print(cars)

# 의미 찾기
car = wordnet.synset('car.n.01')
desc = car.definition()
print(desc)

# 유의어 찾기
lemma = car.lemma_names()
print(lemma)

# 상위어 찾기
hyper = car.hypernym_paths()[0]
print(hyper)

# 유사도 계산하기
dog = wordnet.synset('dog.n.01')
motorcycle = wordnet.synset('motorcycle.n.01')
print(car.path_similarity(dog))
print(car.path_similarity(motorcycle))
