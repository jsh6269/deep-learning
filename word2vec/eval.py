# coding: utf-8
import sys
sys.path.append('..')
from util import most_similar, analogy
import pickle


pkl_file = 'cbow_params.pkl'
# pkl_file = 'skipgram_params.pkl'

with open(pkl_file, 'rb') as f:
    params = pickle.load(f)
    word_vecs = params['word_vecs']
    word_to_id = params['word_to_id']
    id_to_word = params['id_to_word']

# 가장 비슷한(most similar) 단어 뽑기
querys = ['you', 'year', 'car', 'toyota']
for query in querys:
    most_similar(query, word_to_id, id_to_word, word_vecs, top=5)

# 유추(analogy) 작업
print('-'*50)
analogy('king', 'man', 'queen',  word_to_id, id_to_word, word_vecs)
analogy('take', 'took', 'go',  word_to_id, id_to_word, word_vecs)
analogy('car', 'cars', 'child',  word_to_id, id_to_word, word_vecs)
analogy('good', 'better', 'bad',  word_to_id, id_to_word, word_vecs)


'''
[analogy] king:man = queen:?
 woman: 5.5
 mother: 4.98046875
 hacker: 4.66015625
 a.m: 4.59375
 actor: 4.56640625

[analogy] take:took = go:?
 went: 4.40625
 're: 4.29296875
 a.m: 3.853515625
 attended: 3.837890625
 eurodollars: 3.8125

[analogy] car:cars = child:?
 a.m: 6.5234375
 rape: 6.140625
 children: 5.49609375
 incest: 5.04296875
 women: 4.90234375

[analogy] good:better = bad:?
 more: 5.3984375
 rather: 5.30859375
 less: 5.2421875
 greater: 4.171875
 fewer: 3.66796875
'''
