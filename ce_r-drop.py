from __future__ import print_function
import json
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from bert4keras.backend import keras, K
from bert4keras.backend import keras, K, search_layer,batch_gather
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model
from sklearn.model_selection import train_test_split, StratifiedKFold
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, open
from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder
from keras.models import Model
from keras.losses import kullback_leibler_divergence as kld
from keras.callbacks import EarlyStopping, ModelCheckpoint
from rouge import Rouge  # pip install rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert4keras.optimizers import Adam, extend_with_exponential_moving_average
import jieba
from sklearn.model_selection import KFold
jieba.initialize()

# 基本参数
max_c_len = 512
max_t_len = 150
batch_size = 8
epochs = 8
k_sparse = 10
n = 5 
SEED = 2020  

# 模型路径
config_path = '../SPACES-main/datasets/chinese_t5_pegasus_base/config.json'
checkpoint_path = '../SPACES-main/datasets/chinese_t5_pegasus_base/model.ckpt'
dict_path = '../SPACES-main/datasets/chinese_t5_pegasus_base/vocab.txt'


# def load_data(filename):
#     """加载数据
#     单条格式：(标题, 正文)
#     """
#     D = []
#     with open(filename, encoding='utf-8') as f:
#         for l in f:
#             title, content = l.strip().split('\t')
#             D.append((title, content))
#     return D


# # 加载数据集
# train_data = load_data('/root/csl/train.tsv')
# valid_data = load_data('/root/csl/val.tsv')
# test_data = load_data('/root/csl/test.tsv')



def load_data(filename):
    """加载数据
    单条格式：(标题, 正文)
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            id,content, abstract = l.strip().split('|')
            D.append((abstract, content))
    return D

def load_test_data(filename):
    
    D = []
    with open(filename, encoding = 'utf-8') as f:
        for l in f:
            id,content = l.strip().split('|')
            D.append(content)
    return D




train = load_data('train_dataset.csv')
# train_data, val_data = train_test_split(train, shuffle=True, random_state=0, test_size=0.1)
train_data = train[:500]
valid_data = train[501:525]
# valid_data = val_data[:10]
# test_data = val_data[01:350]
# test_data = val_data[501:550]
#22500
test_data = load_test_data('test_dataset.csv')
print(len(train_data))
#2501
print(len(valid_data))




# 构建分词器
tokenizer = Tokenizer(
    dict_path,
    do_lower_case=True,
    pre_tokenize=lambda s: jieba.cut(s, HMM=False)
)


# def generate_copy_labels(source, target):
#     """构建copy机制对应的label
#     """
#     mapping = longest_common_subsequence(source, target)[1]
#     source_labels = [0] * len(source)
#     target_labels = [0] * len(target)
#     i0, j0 = -2, -2
#     for i, j in mapping:
#         if i == i0 + 1 and j == j0 + 1:
#             source_labels[i] = 2
#             target_labels[j] = 2
#         else:
#             source_labels[i] = 1
#             target_labels[j] = 1
#         i0, j0 = i, j
#     return source_labels, target_labels


# def random_masking(token_ids):
#     """对输入进行随机mask，增加泛化能力
#     """
#     rands = np.random.random(len(token_ids))
#     return [
#         t if r > 0.15 else np.random.choice(token_ids)
#         for r, t in zip(rands, token_ids)
#     ]
    

class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_c_token_ids, batch_t_token_ids = [], []
        for is_end, (title, content) in self.sample(random):
            c_token_ids, _ = tokenizer.encode(content, maxlen=max_c_len)
            t_token_ids, _ = tokenizer.encode(title, maxlen=max_t_len)
            
            for i in range(2):
                batch_c_token_ids.append(c_token_ids)
                batch_t_token_ids.append(t_token_ids)
            if len(batch_c_token_ids) == self.batch_size*2 or is_end:
                batch_c_token_ids = sequence_padding(batch_c_token_ids)
                batch_t_token_ids = sequence_padding(batch_t_token_ids)
                yield [batch_c_token_ids, batch_t_token_ids], None
                batch_c_token_ids, batch_t_token_ids = [], []


class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉输入部分
    """
    def compute_loss(self, inputs, mask=None):
        y_true, y_pred = inputs
        alpha = 4
        y_true = y_true[:, 1:]  # 目标token_ids
        y_mask = K.cast(mask[1], K.floatx())[:, 1:]  # 解码器自带mask
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        loss1 = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss1 = K.sum(loss1 * y_mask) / K.sum(y_mask)
        loss2 = kld(y_pred[::2], y_pred[1::2]) + kld(y_pred[1::2], y_pred[::2])
        loss2 = K.sum(loss2 * y_mask[::2]) / K.sum(y_mask[::2])
        
        return  loss1 + loss2 / 4 * alpha
    
    
#     def compute_loss(self, inputs, mask=None):
# #         y_true, y_mask, _, y_pred, _ = inputs
# #         y_true = y_true[:, 1:]  # 目标token_ids
# #         y_mask = y_mask[:, :-1] * y_mask[:, 1:]  # segment_ids，刚好指示了要预测的部分
# #         y_pred = y_pred[:, :-1]  # 预测序列，错开一位
#         y_true, y_pred = inputs
#         y_true = y_true[:,1:]
#         y_mask = K.cast(mask[1],K.floatx())[:,1:]
#         y_pred = y_pred[:,:-1]
#         # 正loss
#         pos_loss = batch_gather(y_pred, y_true[..., None])[..., 0]
#         # 负loss
#         y_pred = tf.nn.top_k(y_pred, k=k_sparse)[0]
#         neg_loss = K.logsumexp(y_pred, axis=-1)
#         # 总loss
#         loss = neg_loss - pos_loss
#         loss = K.sum(loss * y_mask) / K.sum(y_mask)
#         return loss


t5 = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    model='t5.1.1',
    return_keras_model=False,
    name='T5',
)

encoder = t5.encoder
decoder = t5.decoder
model = t5.model
model.summary()

output = CrossEntropy(1)([model.inputs[1], model.outputs[0]])

model = Model(model.inputs, output)


AdamEMA = extend_with_exponential_moving_average(Adam, name='AdamEMA')
optimizer = AdamEMA(learning_rate=2e-5, ema_momentum=0.9999)
model.compile(optimizer=optimizer)
# train_model.summary()



# model.compile(optimizer=Adam(2e-4))




def adversarial_training(model, embedding_name, epsilon=1.):
    """给模型添加对抗训练
    其中model是需要添加对抗训练的keras模型，embedding_name
    则是model里边Embedding层的名字。要在模型compile之后使用。
    """
    if model.train_function is None:  # 如果还没有训练函数
        model._make_train_function()  # 手动make
    old_train_function = model.train_function  # 备份旧的训练函数

    # 查找Embedding层
    for output in model.outputs:
        embedding_layer = search_layer(output, embedding_name)
        if embedding_layer is not None:
            break
    if embedding_layer is None:
        raise Exception('Embedding layer not found')

    # 求Embedding梯度
    embeddings = embedding_layer.embeddings  # Embedding矩阵
    gradients = K.gradients(model.total_loss, [embeddings])  # Embedding梯度
    gradients = K.zeros_like(embeddings) + gradients[0]  # 转为dense tensor

    # 封装为函数
    inputs = (
            model._feed_inputs + model._feed_targets + model._feed_sample_weights
    )  # 所有输入层
    embedding_gradients = K.function(
        inputs=inputs,
        outputs=[gradients],
        name='embedding_gradients',
    )  # 封装为函数

    def train_function(inputs):  # 重新定义训练函数
        grads = embedding_gradients(inputs)[0]  # Embedding梯度
        delta = epsilon * grads / (np.sqrt((grads ** 2).sum()) + 1e-8)  # 计算扰动
        K.set_value(embeddings, K.eval(embeddings) + delta)  # 注入扰动
        outputs = old_train_function(inputs)  # 梯度下降
        K.set_value(embeddings, K.eval(embeddings) - delta)  # 删除扰动
        return outputs

    model.train_function = train_function  # 覆盖原训练函数





class AutoTitle(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        c_encoded = inputs[0]
        return self.last_token(decoder).predict([c_encoded, output_ids])

    def generate(self, text, topk=1):
        c_token_ids, _ = tokenizer.encode(text, maxlen=max_c_len)
        c_encoded = encoder.predict(np.array([c_token_ids]))[0]
        output_ids = self.beam_search([c_encoded], topk=topk)  # 基于beam search
        return tokenizer.decode(output_ids)


autotitle = AutoTitle(
    start_id=tokenizer._token_start_id,
    end_id=tokenizer._token_end_id,
    maxlen=max_t_len
)


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.rouge = Rouge()
        self.smooth = SmoothingFunction().method1
        self.best_bleu = 0.

    def on_epoch_end(self, epoch, logs=None):
        if epoch>5:
            metrics = self.evaluate(valid_data)  # 评测模型
            if metrics['xx'] > self.best_bleu:
                self.best_bleu = metrics['xx']
                model.save_weights('./results/ch_t5_base_best_model_1.weights')  # 保存模型
            metrics['best_bleu'] = self.best_bleu
            print('valid_data:', metrics)

    def evaluate(self, data, topk=3):
        total = 0
        rouge_1, rouge_2, rouge_l, bleu, xx = 0, 0, 0, 0, 0
        for title, content in tqdm(data):
            total += 1
            title = ' '.join(title).lower()
            pred_title = ' '.join(autotitle.generate(content,
                                                     topk=topk)).lower()
            if pred_title.strip():
                scores = self.rouge.get_scores(hyps=pred_title, refs=title)
                rouge_1 += scores[0]['rouge-1']['f']
                rouge_2 += scores[0]['rouge-2']['f']
                rouge_l += scores[0]['rouge-l']['f']
                bleu += sentence_bleu(
                    references=[title.split(' ')],
                    hypothesis=pred_title.split(' '),
                    smoothing_function=self.smooth
                )
                xx = 0.2*rouge_1+0.4*rouge_2+0.4*rouge_l
        rouge_1 /= total
        rouge_2 /= total
        rouge_l /= total
        bleu /= total
        return {
            'rouge-1': rouge_1,
            'rouge-2': rouge_2,
            'rouge-l': rouge_l,
            'bleu': bleu,
            'xx':xx,
        }
    
import pandas as pd
def _predict(data,topk=3):
    pred =[]
    for content in tqdm(data):
        pred_title = ''.join(autotitle.generate(content,
                                               topk=topk)).lower()
        pred.append(pred_title)
        
    sub = pd.read_csv('datasets/test_dataset.csv',sep='|')
    sub['ret'] = pred
    sub = sub[['id','ret']]
    sub.to_csv('sub_t5_base.csv',index=False, sep='|')
    
def __predict(data,topk=3):
    pred =[]
    for content in tqdm(data):
        pred_title = ''.join(autotitle.generate(content,
                                               topk=topk)).lower()
        pred.append(pred_title)
        
        
    sub = pd.DataFrame()
    sub['id'] = [i for i in range(len(pred))]
    sub['ret'] = pred
#     sub = pd.read_csv('datasets/test_dataset.csv',sep='|')
#     sub['ret'] = pred
#     sub = sub[['id','ret']]
    sub.to_csv('results/sample_10.csv',index=False, sep='|')



if __name__ == '__main__':

    evaluator = Evaluator()
    train_generator = data_generator(train_data, batch_size)
#     adversarial_training(model, 'Embedding-Token', 0.5)

    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator]
    )

else:

    model.load_weights('./results/ch_t5_base_best_model_1.weights')

# if __name__ == '__main__':
#     evaluator = Evaluator()
#     model.load_weights('./results/ch_t5_base_best_model_1.weights')
#     __predict(test_data)
    