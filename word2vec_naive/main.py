import sys
sys.path.append('..')

from _common.trainer import Trainer2
from _common.optimizer import Adam
from cbow import CBOW
from skip_gram import SkipGram
from util import preprocess, create_contexts_target, convert_one_hot

window_size = 1
hidden_size = 5
batch_size = 3
max_epoch = 1000

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)

vocab_size = len(word_to_id)
contexts, target = create_contexts_target(corpus, window_size)
target = convert_one_hot(target, vocab_size)
contexts = convert_one_hot(contexts, vocab_size)

model = CBOW(vocab_size, hidden_size)
# model = SkipGram(vocab_size, hidden_size)

optimizer = Adam()
trainer = Trainer2(model, optimizer)

trainer.fit(contexts, target, max_epoch, batch_size)
trainer.plot()

word_vecs = model.word_vecs
for word_id, word in id_to_word.items():
    print(word, word_vecs[word_id])
