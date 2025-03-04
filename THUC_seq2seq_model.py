#! -*- coding: utf-8 -*-
# 法研杯2020 司法摘要
# 生成式：正式模型
# 科学空间：https://kexue.fm

import os, json
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from bert4keras.backend import keras, K, batch_gather
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.optimizers import Adam, extend_with_exponential_moving_average
from bert4keras.snippets import sequence_padding, open
from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder
from bert4keras.snippets import longest_common_subsequence
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from snippets import *
import glob
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# 基本参数
# maxlen = 1024
maxlen = 256
batch_size = 16
epochs = 5
k_sparse = 10
data_seq2seq_json = data_json[:-5] + '_seq2seq.json'
seq2seq_config_json = data_json[:-10] + 'seq2seq_config.json'

if len(sys.argv) == 1:
    fold = 0
else:
    fold = int(sys.argv[1])


def load_data(filename):
    """加载数据
    返回：[{...}]
    """
    D = []
    with open(filename) as f:
        for l in f:
            D.append(json.loads(l))
    return D


if os.path.exists(seq2seq_config_json):
    token_dict, keep_tokens, compound_tokens = json.load(
        open(seq2seq_config_json)
    )
else:
    # 加载并精简词表
    token_dict, keep_tokens = load_vocab(
        dict_path=nezha_dict_path,
        simplified=True,
        startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
    )
    pure_tokenizer = Tokenizer(token_dict.copy(), do_lower_case=True)
    user_dict = []
    for w in load_user_dict(user_dict_path) + load_user_dict(user_dict_path_2):
        if w not in token_dict:
            token_dict[w] = len(token_dict)
            user_dict.append(w)
    compound_tokens = [pure_tokenizer.encode(w)[0][1:-1] for w in user_dict]
    json.dump([token_dict, keep_tokens, compound_tokens],
              open(seq2seq_config_json, 'w'))

tokenizer = Tokenizer(
    token_dict,
    do_lower_case=True,
    pre_tokenize=lambda s: jieba.cut(s, HMM=False)
)


def generate_copy_labels(source, target):
    """构建copy机制对应的label
    """
    mapping = longest_common_subsequence(source, target)[1]
    source_labels = [0] * len(source)
    target_labels = [0] * len(target)
    i0, j0 = -2, -2
    for i, j in mapping:
        if i == i0 + 1 and j == j0 + 1:
            source_labels[i] = 2
            target_labels[j] = 2
        else:
            source_labels[i] = 1
            target_labels[j] = 1
        i0, j0 = i, j
    return source_labels, target_labels


def random_masking(token_ids):
    """对输入进行随机mask，增加泛化能力
    """
    rands = np.random.random(len(token_ids))
    return [
        t if r > 0.15 else np.random.choice(token_ids)
        for r, t in zip(rands, token_ids)
    ]





class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        batch_output_ids, batch_labels = [], []
        for is_end, txt in self.sample(random):
            text = open(txt, encoding='utf-8').read()
            text = text.split('\n')
            if len(text) > 1:
                target = text[0]
                source = '\n'.join(text[1:])
            token_ids, segment_ids = tokenizer.encode(
                source, target, maxlen=maxlen, pattern='S*ES*E'
            )
            idx = token_ids.index(tokenizer._token_end_id) + 1
            masked_token_ids = random_masking(token_ids)
            source_labels, target_labels = generate_copy_labels(
                masked_token_ids[:idx], token_ids[idx:]
            )
            labels = source_labels + target_labels[1:]
            batch_token_ids.append(masked_token_ids)
            batch_segment_ids.append(segment_ids)
            batch_output_ids.append(token_ids)
            batch_labels.append(labels)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_output_ids = sequence_padding(batch_output_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [
                    batch_token_ids, batch_segment_ids, \
                    batch_output_ids, batch_labels
                ], None
                batch_token_ids, batch_segment_ids = [], []
                batch_output_ids, batch_labels = [], []

# class data_generator(DataGenerator):
#     """数据生成器
#     """
#     def __iter__(self, random=False):
#         batch_token_ids, batch_segment_ids = [], []
#         batch_output_ids, batch_labels = [], []
#         for is_end, d in self.sample(random):
#             # i = np.random.choice(2) + 1 if random else 1
#             source, target = d['source_1'], d['target']
#             token_ids, segment_ids = tokenizer.encode(
#                 source, target, maxlen=maxlen, pattern='S*ES*E'
#             )
#             idx = token_ids.index(tokenizer._token_end_id) + 1
#             masked_token_ids = random_masking(token_ids)
#             source_labels, target_labels = generate_copy_labels(
#                 masked_token_ids[:idx], token_ids[idx:]
#             )
#             labels = source_labels + target_labels[1:]
#             batch_token_ids.append(masked_token_ids)
#             batch_segment_ids.append(segment_ids)
#             batch_output_ids.append(token_ids)
#             batch_labels.append(labels)
#             if len(batch_token_ids) == self.batch_size or is_end:
#                 batch_token_ids = sequence_padding(batch_token_ids)
#                 batch_segment_ids = sequence_padding(batch_segment_ids)
#                 batch_output_ids = sequence_padding(batch_output_ids)
#                 batch_labels = sequence_padding(batch_labels)
#                 yield [
#                     batch_token_ids, batch_segment_ids, \
#                     batch_output_ids, batch_labels
#                 ], None
#                 batch_token_ids, batch_segment_ids = [], []
#                 batch_output_ids, batch_labels = [], []

class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉输入部分
    """
    def compute_loss(self, inputs, mask=None):
        seq2seq_loss = self.compute_seq2seq_loss(inputs, mask)
        copy_loss = self.compute_copy_loss(inputs, mask)
        self.add_metric(seq2seq_loss, 'seq2seq_loss')
        self.add_metric(copy_loss, 'copy_loss')
        return seq2seq_loss + 2 * copy_loss

    def compute_seq2seq_loss(self, inputs, mask=None):
        y_true, y_mask, _, y_pred, _ = inputs
        y_true = y_true[:, 1:]  # 目标token_ids
        y_mask = y_mask[:, :-1] * y_mask[:, 1:]  # segment_ids，刚好指示了要预测的部分
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        # 正loss
        pos_loss = batch_gather(y_pred, y_true[..., None])[..., 0]
        # 负loss
        y_pred = tf.nn.top_k(y_pred, k=k_sparse)[0]
        neg_loss = K.logsumexp(y_pred, axis=-1)
        # 总loss
        loss = neg_loss - pos_loss
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss

    def compute_copy_loss(self, inputs, mask=None):
        _, y_mask, y_true, _, y_pred = inputs
        y_mask = K.cumsum(y_mask[:, ::-1], axis=1)[:, ::-1]
        y_mask = K.cast(K.greater(y_mask, 0.5), K.floatx())
        y_mask = y_mask[:, 1:]  # mask标记，减少一位
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss


model = build_transformer_model(
    nezha_config_path,
    nezha_checkpoint_path,
    model='nezha',
    application='unilm',
    with_mlm='linear',
    keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
    compound_tokens=compound_tokens,
)

output = model.get_layer('MLM-Norm').output
output = Dense(3, activation='softmax')(output)
outputs = model.outputs + [output]

# 预测用模型
model = Model(model.inputs, outputs)

# 训练用模型
y_in = Input(shape=(None,))
l_in = Input(shape=(None,))
outputs = [y_in, model.inputs[1], l_in] + outputs
outputs = CrossEntropy([3, 4])(outputs)

train_model = Model(model.inputs + [y_in, l_in], outputs)

AdamEMA = extend_with_exponential_moving_average(Adam, name='AdamEMA')
optimizer = AdamEMA(learning_rate=2e-5, ema_momentum=0.9999)
train_model.compile(optimizer=optimizer)
train_model.summary()


class AutoSummary(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    def get_ngram_set(self, x, n):
        """生成ngram合集，返回结果格式是:
        {(n-1)-gram: set([n-gram的第n个字集合])}
        """
        result = {}
        for i in range(len(x) - n + 1):
            k = tuple(x[i:i + n])
            if k[:-1] not in result:
                result[k[:-1]] = set()
            result[k[:-1]].add(k[-1])
        return result

    @AutoRegressiveDecoder.wraps(default_rtype='logits', use_states=True)
    def predict(self, inputs, output_ids, states):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        prediction = self.last_token(model).predict([token_ids, segment_ids])
        # states用来缓存ngram的n值
        if states is None:
            states = [0]
        elif len(states) == 1 and len(token_ids) > 1:
            states = states * len(token_ids)
        # 根据copy标签来调整概率分布
        probas = np.zeros_like(prediction[0]) - 1000  # 最终要返回的概率分布
        for i, token_ids in enumerate(inputs[0]):
            if states[i] == 0:
                prediction[1][i, 2] *= -1  # 0不能接2
            label = prediction[1][i].argmax()  # 当前label
            if label < 2:
                states[i] = label
            else:
                states[i] += 1
            if states[i] > 0:
                ngrams = self.get_ngram_set(token_ids, states[i])
                prefix = tuple(output_ids[i, 1 - states[i]:])
                if prefix in ngrams:  # 如果确实是适合的ngram
                    candidates = ngrams[prefix]
                else:  # 没有的话就退回1gram
                    ngrams = self.get_ngram_set(token_ids, 1)
                    candidates = ngrams[tuple()]
                    states[i] = 1
                candidates = list(candidates)
                probas[i, candidates] = prediction[0][i, candidates]
            else:
                probas[i] = prediction[0][i]
            idxs = probas[i].argpartition(-k_sparse)
            probas[i, idxs[:-k_sparse]] = -1000
        return probas, states

    def generate(self, text, topk=1):
        max_c_len = maxlen - self.maxlen
        token_ids, segment_ids = tokenizer.encode(text, maxlen=max_c_len)
        output_ids = self.beam_search([token_ids, segment_ids],
                                      topk)  # 基于beam search
        return tokenizer.decode(output_ids)


autosummary = AutoSummary(
    start_id=tokenizer._token_start_id,
    end_id=tokenizer._token_end_id,
    maxlen=maxlen // 2
)

def just_show():
    s1 = u'夏天来临，皮肤在强烈紫外线的照射下，晒伤不可避免，因此，晒后及时修复显得尤为重要，否则可能会造成长期伤害。专家表示，选择晒后护肤品要慎重，芦荟凝胶是最安全，有效的一种选择，晒伤严重者，还请及时就医 。'
    s2 = u'8月28日，网络爆料称，华住集团旗下连锁酒店用户数据疑似发生泄露。从卖家发布的内容看，数据包含华住旗下汉庭、禧玥、桔子、宜必思等10余个品牌酒店的住客信息。泄露的信息包括华住官网注册资料、酒店入住登记的身份信息及酒店开房记录，住客姓名、手机号、邮箱、身份证号、登录账号密码等。卖家对这个约5亿条数据打包出售。第三方安全平台威胁猎人对信息出售者提供的三万条数据进行验证，认为数据真实性非常高。当天下午 ，华 住集 团发声明称，已在内部迅速开展核查，并第一时间报警。当晚，上海警方消息称，接到华住集团报案，警方已经介入调查。'
    for s in [s1, s2]:
        print(u'生成标题:', autosummary.generate(s))
    print()



class Evaluator(keras.callbacks.Callback):
    """训练回调
    """
    def __init__(self):
        self.lowest = 1e10
        self.rouge = Rouge()
        self.smooth = SmoothingFunction().method1
        self.best_bleu = 0.

    def evaluate(self, data, topk=1):
        total = 0
        rouge_1, rouge_2, rouge_l, bleu = 0, 0, 0, 0
        D = []
        for txt in data:
            text = open(txt, encoding='utf-8').read()
            text = text.split('\n')
            if len(text) > 1:
                title = text[0]
                content = '\n'.join(text[1:])
                D.append((title, content))
        for title, content in tqdm(D):
            total += 1
            title = ' '.join(title).lower()
            pred_title = ' '.join(autosummary.generate(content, topk)).lower()
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
        rouge_1 /= total
        rouge_2 /= total
        rouge_l /= total
        bleu /= total
        return {
            'rouge-1': rouge_1,
            'rouge-2': rouge_2,
            'rouge-l': rouge_l,
            'bleu': bleu,
        }
    def on_epoch_end(self, epoch, logs=None):
        optimizer.apply_ema_weights()
        # 保存最优
        metrics = self.evaluate(valid_data)
        # if logs['loss'] <= self.lowest:
        #     self.lowest = logs['loss']
        #     model.save_weights('./best_model.weights')
        if metrics['bleu'] > self.best_bleu:
            self.best_bleu = metrics['bleu']
            model.save_weights('weights/THUC_best_model.weights')  # 保存模型
        metrics['best_bleu'] = self.best_bleu
        print('valid_data:', metrics)
        # 演示效果
        just_show()

        optimizer.reset_old_weights()


if __name__ == '__main__':

    # 加载数据
    # data = load_data(data_seq2seq_json)
    # train_data = data_split(data, fold, num_folds, 'train')
    # valid_data = data_split(data, fold, num_folds, 'valid')
    # train_data = load_data("/home/transwarp/gujiasheng/csl_title_public/csl_title_train_seq2seq.json")
    # train_valid = load_data("/home/transwarp/gujiasheng/csl_title_public/csl_title_dev_seq2seq.json")
    txts = glob.glob('/home/transwarp/gujiasheng/data/THUCNews/*/*.txt')
    train_data = txts[:int(0.999 * len(txts))]
    valid_data = txts[int(0.999 * len(txts)):]

    # 启动训练
    evaluator = Evaluator()
    train_generator = data_generator(train_data, batch_size)
    # model.load_weights('weights/seq2seq_model.1.weights')
    train_model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator]
    )

else:

    # model.load_weights('weights/seq2seq_model_wonezha_THUC.%s.weights' % (epochs - 1))
    model.load_weights('weights/THUC_best_model.weights')
