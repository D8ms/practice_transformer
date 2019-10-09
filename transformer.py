import glob, os
import tensorflow as tf
from keras import optimizers
import numpy as np


class SelfAttention:
    def __init__(self, inp, emb_size, num_tokens, num_head, mask=None):
        #make sure the shape is [batch, token, embedding]
    
        self.queries = Dense(emb_size * num_head)(inp)
        self.keys = Dense(emb_size * num_head)(inp)
        self.values = Dense(emb_size * num_head)(inp)
        #shape should be [batch, token, embedding * num_head]

        _queries = tf.reshape(self.queries, [-1, num_tokens, emb_size, num_heads])
        _queries = tf.transpose(_queries, [0, 3, 1, 2])
        _queries = tf.reshape(_queries, [-1, num_tokens, emb_size])


        _keys = tf.reshape(self.keys, [-1, num_tokens, emb_size, num_heads])
        _keys = tf.transpose(_keys, [0, 3, 2, 1])
        _keys = tf.reshape(_keys, [-1, emb_size, num_tokens])

        _values = tf.reshape(self.values, [-1, num_tokens, emb_size, num_heads])
        _values = tf.transpose(_values, [-1, 3, 1, 2])
        _values = tf.reshape(_values, [-1, num_tokens, emb_size])

        compatability = tf.matmul(transposed_queries, transposed_keys)
        scaled_compat = tf.div(compatability, tf.sqrt(emb_size))

        if mask:
            scaled_compat -= (mask * 1e9)
        
        softmaxed_compat = tf.softmax(scaled_compat, axis=-1)
        #[batch * head, emb_size, emb_size

        weighted_values = tf.matmul(_values, softmaxed_compat)
        weighted_values = reshape(weighted_values, [-1, num_head, emb_size, emb_size])
        weighted_values = tf.transpose(weighted_values, [0, 2, 3, 1])
        weighted_values = reshape(weighted_values, [-1, emb_size, emb_size * num_heads])

        #size should be [batch, num_tokens]
        #mabe mask here

        unified_values = Dense(emb_size)(weighted_values) 
        return unified_values

class Transformer:
    def __init__(self, inp, emb_size, num_tokens, num_head, mlp_multiplier=4, mask=None):
        
        attentioned = SelfAttention(inp, emb_size, num_tokens, num_head, mask=None)
        summed = inp + attentioned
        normed = LayerNormalization(epsilon=1e-6)(summed)
        mlp_residual = normed + self.smallMLP(normed, emb_size, mlp_multiplier)
        return LayerNormalization(epsilon=1e-6)(mlp_residual)

    def smallMLP(self, x, emb_size, multiplier):
        x = Dense(emb_size * multiplier, activation=tf.nn.ReLU)(x)
        x = Dense(emb_size * multiplier, activation=tf.nn.ReLU)(x)
        x = Dense(emb_size * multiplier, activation=tf.nn.ReLU)(x)
        return Dense(emb_size, activation=tf.nn.ReLU)(x)

class SentimentClassifier:
    def __init__(self, vocab_size, emb_size, num_tokens, num_head, batch_size=25 mlp_multiplier=4):
        self.load_data(vocab_size, num_tokens)

        self.tokenizer.fit_on_texts(self.train_pos + self.train_neg)
        self.createModel(vocab_size, emb_size, num_tokens, num_heads, mlp_multiplier)

    def load_data(self, vocab_size, maxlen):
        train_pos = self.load_dir("imdb/aclImdb/train/pos/")
        train_neg = self.load_dir("imdb/aclImdb/train/neg/")
        test_pos = self.load_dir("imdb/aclImdb/test/pos/")
        test_neg = self.load_dir("imdb/aclImdb/test/neg/")

        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(
            num_words = vocab_size - 1,
            lower = True,
            oov_token = 'unk'
        )

        self.train_pos = np.array(pad_sequence(self.tokenizer.texts_to_sequences(train_pos), maxlen))
        self.train_neg = np.array(pad_sequence(self.tokenizer.texts_to_sequences(train_neg), maxlen))
        self.test_pos =  np.array(pad_sequence(self.tokenizer.texts_to_sequences(test_pos),  maxlen))
        self.test_neg =  np.array(pad_sequence(self.tokenizer.texts_to_sequences(test_neg),  maxlen))

    def get_batch(self, batch_size):
        #assuming we have equal numebr of pos and neg
        num_pos = np.random.randint(1, batch_size)
        num_neg = batch_size - num_pos
        pos_indices = np.random.randint(0, len(self.train_pos), num_pos) 
        neg_indices = np.random.randint(0, len(self.train_pos), num_neg) 
        
        data = np.concatenate((self.train_pos[pos_indices], self.train_neg[neg_indices])
        targets = np.array([[1.0, 0.0]] * num_pos + [[0.0, 1.0]] * num_neg)

        return data, targets

    def load_dir(self, dirpath):
        ret = []
        for filename in glob.glob(dirpath + "*.txt"):
            with open(file, 'r') as fh:
                ret.append(fh.read())
        return ret
    
    def positionalEncoding(self, num_tokens, emb_size):
        angles = self.getAngles(
            np.arange(num_tokens)[:, None],
            np.arange(emb_size)[None, :],
            emb_size
        )

        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.sin(angles[:, 1::2])

        return tf.cast(angles, dtype=tf.float32)
        
    def getAngles(self, pos, i, emb_dim):
        return pos / (np.power(10000, (2 * (i//2)) / np.float32(emb_dim))
    
    def padMask(self, tokens, pad_value=0):
        #assume 0 are padding and every other token is positive
        pad_mask = np.zeros(tokens.shape)
        pad_mask[tokens == pad_value] = 1
        return pad_mask

    def createModel(self, vocab_size, emb_size, num_tokens, num_head, mlp_multiplier=4):
        embedding = Embedding(vocab_size, emb_size, input_length = num_tokens)
        positional_encoding = self.positionalEncoding(num_tokens, emb_size)

        self.input_ph = tf.placeholder(np.int64, shape=[None, num_tokens], name="token_indexes_ph")
        self.pad_mask_ph = tf.placeholder(np.float32, shape=[None, num_tokens], name="pad_mask_ph")
        self.target_ph = tf.placeholder(np.float32, shape=[None, 2], name="target_ph")
        input = tf.convert_to_tensor(self.input)
        target = tf.convert_to_tensor(self.target_ph)
        embedding = embedding(input) + positional_encoding
        
        t1 = Transformer(embedding, emb_size, num_tokens, num_head, self.pad_mask_ph)
        t2 = Transformer(t1, emb_size, num_tokens, num_head)
        t3 = Transformer(t2, emb_size, num_tokens, num_head)
        avg = tf.mean(t3, axis=-1)
        self.prediction = Dense(2, activation=tf.nn.softmax)(avg)
        
        self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=target, logits=self.prediction)
        self.train_op = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

    def batchTrain(self, sess, batch_size=25):
        batch_data, batch_taget = self.get_batch(batch_size):
        pad_mask = self.padMask(batch_data)
        sess.run(
            [self.train_op, self.loss],
            feed_dict = {
                self.input_ph: batch_data,
                self.pad_mask_ph: pad_mask,
                self.target_ph: batch_target
            }
        )

def Generator:
    def positionalEncoding(self, num_tokens, emb_size):
        angles = self.getAngles(
            np.arange(num_tokens)[:, None],
            np.arange(emb_size)[None, :],
            emb_size
        )

        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.sin(angles[:, 1::2])

        return tf.cast(angles, dtype=tf.float32)
        
    def getAngles(self, pos, i, emb_dim):
        return pos / (np.power(10000, (2 * (i//2)) / np.float32(emb_dim))
         
        
    def upperTriangleMatrix(self, size):
        return 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    

SentimentClassifier(10000, 128, 512, 
    def __init__(self, vocab_size, emb_size, num_tokens, num_head, batch_size=25 mlp_multiplier=4):
