#! -*- coding: utf-8 -*-
# 法研杯2020 司法摘要
# 生成式：数据转换
# 科学空间：https://kexue.fm

# from extract_model import *
from bert4keras.snippets import open
import json
import numpy as np
from tqdm import tqdm
from snippets import *
def load_data(filename):
    """加载数据
    返回：[(texts, labels, summary)]
    """
    D = []
    with open(filename) as f:
        for l in f:
            D.append(json.loads(l))
    return D


def fold_convert(data, data_x, fold):
    """每一fold用对应的模型做数据转换
    """
    valid_data = data_split(data, fold, num_folds, 'valid')
    valid_x = data_split(data_x, fold, num_folds, 'valid')

    model.load_weights('weights/extract_model.%s.weights' % fold)
    y_pred = model.predict(valid_x)[:, :, 0]

    results = []
    for d, yp in tqdm(zip(valid_data, y_pred), desc=u'转换中'):
        yp = yp[:len(d[0])]
        yp = np.where(yp > threshold)[0]
        source_1 = ''.join([d[0][i] for i in yp])
        source_2 = ''.join([d[0][i] for i in d[1]])
        result = {
            'source_1': source_1,
            'source_2': source_2,
            'target': d[2],
        }
        results.append(result)

    return results
def fold_convert0(data,fold):
    """每一fold用对应的模型做数据转换
    """
    valid_data = data_split(data, fold, num_folds, 'valid')
    # valid_x = data_split(data_x, fold, num_folds, 'valid')

    # model.load_weights('weights/extract_model.%s.weights' % fold)
    # y_pred = model.predict(valid_x)[:, :, 0]

    results = []
    for d in tqdm(valid_data, desc=u'转换中'):
        # yp = yp[:len(d[0])]
        # yp = np.where(yp > threshold)[0]
        # source_1 = ''.join([d[0][i] for i in yp])
        # source_2 = ''.join([d[0][i] for i in d[1]])
        source_1 = d["abst"]
        result = {
            'source_1': source_1,
            # 'source_2': source_2,
            'target': d["title"],
        }
        results.append(result)

    return results

def convert(filename, data):
# def convert(filename, data, data_x):
    """转换为生成式数据
    """
    F = open(filename, 'w', encoding='utf-8')
    total_results = []
    for fold in range(num_folds):
        total_results.append(fold_convert0(data, fold))
        # total_results.append(fold_convert(data, data_x, fold))

    # 按照原始顺序写入到文件中
    n = 0
    while True:
        i, j = n % num_folds, n // num_folds
        try:
            d = total_results[i][j]
        except:
            break
        F.write(json.dumps(d, ensure_ascii=False) + '\n')
        n += 1

    F.close()


if __name__ == '__main__':

    # data = load_data(data_extract_json)
    csl_train_json = '/home/transwarp/gujiasheng/csl_title_public/csl_title_dev.json'
    data = load_data(csl_train_json)
    # data_x = np.load(data_extract_npy)
    data_seq2seq_json = csl_train_json[:-5]+ '_seq2seq.json'
    # data_seq2seq_json = data_json[:-5] + '_seq2seq.json'
    convert(data_seq2seq_json, data)
    print(u'输出路径：%s' % data_seq2seq_json)
