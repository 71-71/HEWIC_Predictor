"""
This file performs data processing.
Date: 2022-07-30
"""
import pandas as pd
import numpy as np
import calendar
import jieba
import math
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

def preprocess_data_long(data_long_file):
    long_df = pd.read_csv(data_long_file)
    long_df['date'] = pd.to_datetime(long_df['date'])
    dataset = []
    masks = []
    max_seq_len = 31
    start_date = pd.to_datetime({'year':[2022], 'month':[3], 'day':[1]})
    def get_days(x):
        return (x-start_date).dt.days
    columns_val = long_df.columns[2:]
    mode = 'fixed'
    for gid, gdf in long_df.groupby(['Sample ID']):
        gdf['delta'] = gdf['date'].apply(get_days)
        feature = gdf[columns_val].values
        day_index = gdf['delta'].tolist()
        day_index = [x-1 for x in day_index]
        c_feature = np.zeros((max_seq_len,feature.shape[1]))
        c_feature[day_index,:] = feature
        dataset.append(c_feature)
        if mode=='fixed':
            masks.append([1]*max_seq_len)
        else:
            year = start_date.year
            month = start_date.month
            cur_len = calendar.monthrange(year, month)[1]
            masks.append([1]*cur_len+(max_seq_len-cur_len)*[0])
    np.savez("data/data_long.npz", datas=np.array(dataset), mask=np.array(masks)[:,:,np.newaxis])

def preprocess_data_cs(data_cs_file):
    cs_df = pd.read_csv(data_cs_file)
    cs_columns = cs_df.columns[1:]
    cs_df_data = cs_df[cs_columns]
    def min_max_norm(data,col_names):
        for col in col_names:
            data[col] = (data[col]-data[col].min())/(data[col].max()-data[col].min())
        return data
    cs_df_data = min_max_norm(cs_df_data,['height','weight','Initial dialysis age'])
    np.save("data/cs_data",cs_df_data.values)

def preprocess_data_text(train_corpus_file,data_text_file):
    def get_train_test_corpus(train_text_df,target_text_df):

        train_symptom_docs = []
        train_result_docs = []
        for i, text in enumerate(train_text_df['symptom'].values):
            if  not isinstance(text,str):
                word_list=[]
            else:
                word_list = list(jieba.cut(text)) #分词
            document = TaggedDocument(word_list, tags=[i])
            train_symptom_docs.append(document)
        for i, text in enumerate(train_text_df['result'].values):
            if  not isinstance(text,str):
                word_list=[]
            else:
                word_list = list(jieba.cut(text)) #分词
            document = TaggedDocument(word_list, tags=[i])
            train_result_docs.append(document)
        target_text_symtom, target_text_result = [], []
        for i, text in enumerate(target_text_df['symptom'].values):
            if  not isinstance(text,str):
                word_list=[]
            else:
                word_list = list(jieba.cut(text))
            target_text_symtom.append(word_list)
        for i, text in enumerate(target_text_df['result'].values):
            if  not isinstance(text,str):
                word_list = []
            else:
                word_list = list(jieba.cut(text))
            target_text_result.append(word_list)
    
        return train_symptom_docs, train_result_docs, target_text_symtom, target_text_result
    #模型训练 所有数据
    def train_and_test(train_corpus, target_corpus_symtom, target_corpus_result, size=50, epoch_num=50,name=None):#每条记录变成50维向量
        model_dm = Doc2Vec(train_corpus, min_count=1, window=3, vector_size=size, sample=1e-3, negative=5, workers=4)
        model_dm.train(train_corpus, total_examples=model_dm.corpus_count, epochs=epoch_num)
        embeds_symptom, embeds_result = [],[]
        for sentence in target_corpus_symtom:
            inferred_vector_dm = model_dm.infer_vector(sentence)
            embeds_symptom.append(inferred_vector_dm)
        for sentence in target_corpus_result:
            inferred_vector_dm = model_dm.infer_vector(sentence)
            embeds_result.append(inferred_vector_dm)
        return np.concatenate([np.array(embeds_symptom),np.array(embeds_result)],axis=-1)
    train_text_df = pd.read_csv(train_corpus_file)
    target_text_df = pd.read_csv(data_text_file)
    train_symptom_docs, train_result_docs, target_text_symtom, target_text_result = \
    get_train_test_corpus(train_text_df,target_text_df)
    embs = train_and_test(train_symptom_docs+train_result_docs,target_text_symtom, target_text_result)
    np.save('data/text_emb',embs)
    

if __name__ == "__main__":

    preprocess_data_long('data/case_data_long.csv')
    preprocess_data_cs('data/case_data_cs.csv')
    preprocess_data_text('data/train_corpus.csv','data/case_data_text.csv')

    labels_1m = np.array([1,1,0,0,0,1,1,0]).reshape(-1,1)
    np.save('data/y_1month',labels_1m)
    labels_3m = np.array([1,1,1,0,0,1,1,0]).reshape(-1,1)
    np.save('data/y_3month',labels_3m)

    