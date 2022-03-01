from __future__ import print_function
import numpy as np
from tqdm import tqdm
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, open
from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder
from keras.models import Model
from rouge import Rouge  # pip install rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


from sklearn.model_selection import train_test_split, StratifiedKFold
# 基本参数
maxlen = 500
batch_size = 16
epochs = 10

# bert配置
# config_path = './chinese_rbt_ext/bert_config.json'
# checkpoint_path = './chinese_rbt_ext/bert_model.ckpt'
# dict_path = './chinese_rbt_ext/vocab.txt'

config_path = '../tianchi_qg/qg/user_data/model_data/bert_config.json'
checkpoint_path = '../tianchi_qg/qg/user_data/model_data/model.ckpt-691689'
dict_path = '../tianchi_qg/qg/user_data/model_data/vocab.txt'


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


# 加载数据集
train = load_data('train_dataset.csv')
train_data, valid_data = train_test_split(train, shuffle=True, random_state=0, test_size=0.03)
# train_data = train_data[:200]
valid_data = valid_data[:500]


# valid_data = load_data('csl_title_public/dev.tsv')
test_data = load_test_data('test_dataset.csv')
test_data = test_data[:10]
# test_data = test_data[:20]

# 加载并精简词表，建立分词器
token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        for is_end, (title, content) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(
                content, title, maxlen=maxlen
            )
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [batch_token_ids, batch_segment_ids], None
                batch_token_ids, batch_segment_ids = [], []


class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉输入部分
    """
    def compute_loss(self, inputs, mask=None):
        y_true, y_mask, y_pred = inputs
        y_true = y_true[:, 1:]  # 目标token_ids
        y_mask = y_mask[:, 1:]  # segment_ids，刚好指示了要预测的部分
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss


model = build_transformer_model(
    config_path,
    checkpoint_path,
    model='nezha',
    application='unilm',
    keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
)

output = CrossEntropy(2)(model.inputs + model.outputs)

model = Model(model.inputs, output)
model.compile(optimizer=Adam(1e-5))
model.summary()




class AutoTitle(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        return self.last_token(model).predict([token_ids, segment_ids])

    def generate(self, text, topk=1):
        max_c_len = maxlen - self.maxlen
        token_ids, segment_ids = tokenizer.encode(text, maxlen=max_c_len)
        output_ids = self.beam_search([token_ids, segment_ids],
                                      topk=topk)  # 基于beam search
        return tokenizer.decode(output_ids)


autotitle = AutoTitle(start_id=None, end_id=tokenizer._token_end_id, maxlen=100)


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.rouge = Rouge()
        self.smooth = SmoothingFunction().method1
        self.best_bleu = 0.

    def on_epoch_end(self, epoch, logs=None):
#         if epoch<5:
#             continue
# #         if epoch%5==4:
        if epoch>5 and epoch%3==0:
            metrics = self.evaluate(valid_data)  # 评测模型
            if metrics['xx'] > self.best_bleu:
                self.best_bleu = metrics['xx']
                model.save_weights('./best_model.weights')  # 保存模型
            metrics['best_belu'] = self.best_bleu
            print('valid_data:', metrics)
            
            
 

    def evaluate(self, data, topk=1):
        total = 0
        rouge_1, rouge_2, rouge_l, bleu,xx = 0, 0, 0, 0,0
        for title, content in tqdm(data):
            total += 1
            title = ' '.join(title).lower()
            pred_title = ' '.join(autotitle.generate(content, topk)).lower()
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
                xx = 0.2*rouge_1 + 0.4*rouge_2 +0.4*rouge_l
        rouge_1 /= total
        rouge_2 /= total
        rouge_l /= total
        bleu /= total
        return {
            'rouge-1': rouge_1,
            'rouge-2': rouge_2,
            'rouge-l': rouge_l,
            'bleu': bleu,
            'xx':xx
        }

    
    
import pandas as pd


# def _predict(data,topk=1):
#     pred =[]
#     for content in tqdm(data):
#         pred_title = ''.join(autotitle.generate(content,
#                                                topk=topk)).lower()
#         pred.append(pred_title)
        
#     sub = pd.read_csv('test_dataset.csv',sep='|')
#     sub['ret'] = pred
#     sub = sub[['id','ret']]
#     sub.to_csv('sub_unilm_103.csv',index=False, sep='|')
    
def _predict(data,topk=1):
    pred = []
    for content in tqdm(data):
        pred_title = ''.join(autotitle.generate(content,
                                               topk=topk)).lower()
        pred.append(pred_title)
        
    sub = pd.DataFrame()
    
    sub['id'] = [i for i in range(len(pred))]
    sub['ret'] = pred
    sub.to_csv('sub.csv',index=False, sep='|')

        

# 训练
# if __name__ == '__main__':
    

#     evaluator = Evaluator()
#     train_generator = data_generator(train_data, batch_size)

#     model.fit(
#         train_generator.forfit(),
#         steps_per_epoch=len(train_generator),
#         epochs=epochs,
#         callbacks=[evaluator]
#     )

# else:

#     model.load_weights('./best_model.weights')
    
# # 测试 
if __name__ == '__main__':
    evaluator = Evaluator()
    model.load_weights('best_model.weights')
    _predict(test_data)
    