import glob, os
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, Embedding, LayerNormalization
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, emb_size, num_heads, num_tokens):
        #make sure the shape is [batch, token, embedding]
        super(SelfAttention, self).__init__()
        self.num_tokens = num_tokens
        self.emb_size = emb_size
        self.num_heads = num_heads

        self.queries_dense = Dense(emb_size * num_heads, use_bias = False)
        self.keys_dense = Dense(emb_size * num_heads, use_bias = False)
        self.values_dense = Dense(emb_size * num_heads, use_bias = False)
        self.unify_dense = Dense(emb_size)
        #shape should be [batch, token, embedding * num_heads]

    def call(self, inp, mask=None):
        q = self.queries_dense(inp)
        k = self.keys_dense(inp)
        v = self.values_dense(inp)
        return self.unify_dense(self.getScaledValues(q, k, v, mask))

    def getScaledValues(self, queries, keys, values, mask):

        _queries = tf.reshape(queries, [-1, self.num_tokens, self.emb_size, self.num_heads])
        _queries = tf.transpose(_queries, [0, 3, 1, 2])


        _keys = tf.reshape(keys, [-1, self.num_tokens, self.emb_size, self.num_heads])
        _keys = tf.transpose(_keys, [0, 3, 2, 1])

        _values = tf.reshape(values, [-1, self.num_tokens, self.emb_size, self.num_heads])
        _values = tf.transpose(_values, [0, 3, 1, 2])

        #batch*head, num_tokens, emb_size * batch*head, emb_size, num_token]
        compatability = tf.matmul(_queries, _keys)
        #[batch, head, num_tokens, num_tokens]
        scaled_compat = tf.divide(compatability, tf.sqrt(tf.cast(self.emb_size, tf.float32)))

        if mask is not None:
            scaled_compat -= (mask * 1e9)
        
        softmaxed_compat = tf.nn.softmax(scaled_compat, axis=-1)
        #shape = batch, num_head, emb_size, emb_size
        
        #[batch, head, token, token] * [batch, head, token, es]
        #expected: batch, head, token, es
        weighted_values = tf.matmul(softmaxed_compat, _values)
        #assert(weighted_values.shape[1:] == [self.num_heads, self.num_tokens, self.emb_size])
        weighted_values = tf.transpose(weighted_values, [0, 2, 3, 1])
        #assert(weighted_values.shape[1:] == [self.num_tokens, self.emb_size, self.num_heads])
        weighted_values = tf.reshape(weighted_values, [-1, self.num_tokens, self.emb_size * self.num_heads])

        return weighted_values

class Transformer(tf.keras.layers.Layer):
    def __init__(self, emb_size, num_tokens, num_heads, mlp_multiplier=4):
        super(Transformer, self).__init__()
        self.attention = SelfAttention(emb_size, num_heads, num_tokens)
        self.mlpDense1 = Dense(emb_size * mlp_multiplier, activation=tf.nn.relu)
        #self.mlpDense2 = Dense(emb_size * mlp_multiplier, activation=tf.nn.relu)
        #self.mlpDense3 = Dense(emb_size * mlp_multiplier, activation=tf.nn.relu)
        self.mlpDense4 = Dense(emb_size, activation=None)
       
    def call(self, inp, mask=None):
        unified_values = self.attention(inp, mask)
        summed = inp + unified_values
        normed = LayerNormalization(epsilon=1e-6)(summed)
        mlp_residual = normed + self.mlp(normed)
        return LayerNormalization(epsilon=1e-6)(mlp_residual)

    def mlp(self, x):
        x = self.mlpDense1(x)
        #x = self.mlpDense2(x)
        #x = self.mlpDense3(x)
        return self.mlpDense4(x)

class SentimentClassifier(tf.keras.Model):
    def __init__(self, vocab_size, emb_size, num_tokens, num_heads, batch_size=25, mlp_multiplier=4):
        super(SentimentClassifier, self).__init__()
        self.load_data(vocab_size, num_tokens)

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
        self.tokenizer.fit_on_texts(train_pos + train_neg)
        self.train_pos = np.array(pad_sequences(self.tokenizer.texts_to_sequences(train_pos), maxlen))
        self.train_neg = np.array(pad_sequences(self.tokenizer.texts_to_sequences(train_neg), maxlen))
        self.test_pos =  np.array(pad_sequences(self.tokenizer.texts_to_sequences(test_pos),  maxlen))
        self.test_neg =  np.array(pad_sequences(self.tokenizer.texts_to_sequences(test_neg),  maxlen))

    def getBatchData(self, batch_size):
        #assuming we have equal numebr of pos and neg
        num_pos = np.random.randint(1, batch_size)
        num_neg = batch_size - num_pos
        pos_indices = np.random.randint(0, len(self.train_pos), num_pos) 
        neg_indices = np.random.randint(0, len(self.train_pos), num_neg) 
        
        data = np.concatenate((self.train_pos[pos_indices], self.train_neg[neg_indices]))
        targets = np.array([[1.0, 0.0]] * num_pos + [[0.0, 1.0]] * num_neg)

        return data, targets
    
    def getRiggedBatchData(self, batch_size):
        #assuming we have equal numebr of pos and neg
        num_pos = batch_size // 2
        num_neg = batch_size - num_pos
        pos_indices = np.arange(num_pos) 
        neg_indices = np.arange(num_neg)
        
        data = np.concatenate((self.train_pos[pos_indices], self.train_neg[neg_indices]))
        targets = np.array([[1.0, 0.0]] * num_pos + [[0.0, 1.0]] * num_neg)

        return data, targets

    def load_dir(self, dirpath):
        ret = []
        for filename in glob.glob(dirpath + "*.txt"):
            with open(filename, 'r') as fh:
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
        return pos / (np.power(10000, (2 * (i//2)) / np.float32(emb_dim)))
    
    def padMask(self, tokens, pad_value=0):
        #assume 0 are padding and every other token is positive
        pad_mask = np.zeros(tokens.shape)
        pad_mask[tokens == pad_value] = 1
        return pad_mask[:, None, None, :]

    def getLoss(self, inp, pad_mask, target):
        prediction = self.infer(inp, pad_mask) 
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=prediction))
        return prediction, loss

    def infer(self, inp, pad_mask):
        combined_input = self.embedding(inp) + self.positional_encoding
        x = self.t1(combined_input, pad_mask)
        x = self.t2(x)
        x = self.t3(x)
        
        avg = tf.reduce_mean(x, axis=1)
        prediction = self.class_dense(avg)

        return prediction

    def createModel(self, vocab_size, emb_size, num_tokens, num_heads, mlp_multiplier=4):
        self.embedding = Embedding(vocab_size, emb_size, input_length = num_tokens)
        self.positional_encoding = self.positionalEncoding(num_tokens, emb_size)
        
        self.class_dense = Dense(2)
        
        self.t1 = Transformer(emb_size, num_tokens, num_heads)
        self.t2 = Transformer(emb_size, num_tokens, num_heads)
        self.t3 = Transformer(emb_size, num_tokens, num_heads)
        
    def getBatchLoss(self, batch_size=25):
        batch_data, batch_target = self.getBatchData(batch_size)
        pad_mask = self.padMask(batch_data)
        return self.getLoss(batch_data, pad_mask, batch_target) #prediction, loss

class Generator:
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
        return pos / (np.power(10000, (2 * (i//2)) / np.float32(emb_dim)))
         
        
    def upperTriangleMatrix(self, size):
        return 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)


optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
epoch = 0
sentiment_classifier = SentimentClassifier(10000, 128, 512, 8, 25, 4)
while epoch < 1000000:
    with tf.GradientTape() as tape:
        prediction, loss = sentiment_classifier.getBatchLoss()
    print("epoch", epoch)
    print("loss", loss)
    
    gradients = tape.gradient(loss, sentiment_classifier.trainable_variables)
    optimizer.apply_gradients(zip(gradients, sentiment_classifier.trainable_variables))
    epoch += 1
        
