import sys
sys.path.append('..')
from sklearn.utils.extmath import randomized_svd

from util import most_similar, create_co_matrix, ppmi
from _dataset import ptb


window_size = 2
wordvec_size = 100

corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)

print('동시발생 수 계산 ...')
C = create_co_matrix(corpus, vocab_size, window_size)

print('PPMI 계산 ...')
W = ppmi(C, verbose=True)

print('calculating SVD ...')
U, S, V = randomized_svd(W, n_components=wordvec_size, n_iter=5, random_state=None)

word_vecs = U[:, :wordvec_size]

querys = ['you', 'year', 'car', 'toyota']
for query in querys:
    most_similar(query, word_to_id, id_to_word, word_vecs, top=5)


'''
[query] you
 i: 0.6624540090560913
 we: 0.6131225824356079
 do: 0.6008265018463135
 anybody: 0.5518854856491089
 something: 0.49773815274238586

[query] year
 month: 0.6794378757476807
 quarter: 0.6311956644058228
 last: 0.6147191524505615
 next: 0.6048276424407959
 february: 0.5880733728408813

[query] car
 auto: 0.6559597253799438
 luxury: 0.633526623249054
 cars: 0.524562418460846
 truck: 0.4893079698085785
 domestic: 0.480075865983963

[query] toyota
 motor: 0.6908928751945496
 motors: 0.6681710481643677
 nissan: 0.6354405879974365
 honda: 0.5952098965644836
 mazda: 0.538728654384613
'''
