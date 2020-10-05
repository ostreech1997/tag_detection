import textdistance
import numpy as np
from deeppavlov.core.common.file import read_json
from deeppavlov import build_model, configs
import os 
import json
import pandas as pd 
import re 
from collections import Counter
import pickle
from pydoc import locate
import lib_new.common as cm
# import stanfordnlp
# from spacy_stanfordnlp import StanfordNLPLanguage

class TagDetection:
    def __init__(self, path_to_embedder_model: str = ''):
        self.embedder = False
        self.lem = False
        if path_to_embedder_model:
            self.path_to_embedder_model = path_to_embedder_model

    def load_embedder_model_mult(self):
        # Импортимруем предтренерованный embedder 
        bert_config = read_json(configs.embedder.bert_embedder)
        bert_config['metadata']['variables']['BERT_PATH'] = f'{self.path_to_embedder_model}/multi_cased_L-12_H-768_A-12_pt'
        self.embedder = build_model(bert_config, download=False)
        
    def load_embedder_model_ru(self):
        # Импортимруем предтренерованный embedder 
        bert_config = read_json(configs.embedder.bert_embedder)
        bert_config['metadata']['variables']['BERT_PATH'] = f'{self.path_to_embedder_model}/rubert_cased_L-12_H-768_A-12_pt'
        self.embedder = build_model(bert_config, download=False)    

    def get_embedder_model(self, emb_type):
        if not self.embedder:
            if emb_type == 'mult':
                self.load_embedder_model_mult()
            elif emb_type == 'ru':
                self.load_embedder_model_ru()
        return self.embedder

    def load_lem_model(self):
        stanfordnlp = locate('stanfordnlp')
        StanfordNLPLanguage = locate('spacy_stanfordnlp.StanfordNLPLanguage')
        snlp = stanfordnlp.Pipeline(lang='ru')
        self.lem = StanfordNLPLanguage(snlp)

    def get_lem_model(self):
        if not self.lem:
            self.load_lem_model()
        return self.lem

    # l - list of str
    def lemmatization(self, l):
        doc = [self.get_lem_model()(a) for a in l]
        doc_lemma = [d[0].lemma_ for d in doc]
        return doc_lemma

    # Создает словарь для токенов в файл. Он используется для расчета коэффициента tf-idf
    def cnt_dict(self, df, save_path, emb_type):
        df['original_name'] = df['name']
        df['word_emb'] = df['name'].apply(lambda x: self.get_embedder_model(emb_type)([x])[0][0])
        # лематизация
        df['word_emb'] = df['word_emb'].apply(lambda x: self.lemmatization(x))
        # Переводим их в лист
        word_emb = [l for list_ in df['word_emb'].tolist() for l in list_]
        # Заполняем словарь токенов, где ключ токен - значение кол-во документов, где есть токен
        cnt = Counter()
        for word in word_emb:
            cnt[word] += 1
        a_file = open(save_path, "wb")
        pickle.dump(cnt, a_file)
        a_file.close()
        return cnt

    # Из n векторов получаем один вектор (среднее)
    def vecs2vec_avg(self, vecs):
        return np.sum(vecs,axis=0,keepdims=True)/vecs.shape[1]

    # Из n векторов получаем один вектор веса tf_idf
    def vecs2vec_tfidf(self, tag_tokens, tag_vecs, cnt):
        num_tokens = len(tag_vecs) # количество токенов в запросе после NER 
        tf_idf_list = [] # list с коэффицентами для tf-idf
        for i in range(num_tokens):
            # k - число документов в которых есть данный токен
            k = cnt[tag_tokens[i]] # k = 0 если нет слвоа в словаре
            # если до этого слово не встречалось -> добавляем его 
            if k == 0:
                cnt[tag_tokens[i]] += 1
                k += 1
            N = len(cnt)
            tf_i = 1/num_tokens
            idf_i = np.log(N/k)
            tf_idf_i = tf_i * idf_i
            tf_idf_list.append(tf_idf_i)

        # отнормируем сумму весов к 1 
        tf_idf_list = tf_idf_list/np.sum(tf_idf_list, axis=0).reshape(1,1)
        # взвесили вектора токенов с помощью tf_idf
        vec_tfidf = tag_vecs * tf_idf_list.T
        # получаем итоговый вектор
        tag_vec_tfidf = np.sum(vec_tfidf, axis=0, keepdims=True) # Итоговый веткор (1,768)
        return tag_vec_tfidf

    def tag2vec(self, tag, method='avg', cnt={}, emb_type='mult'):
        tag_embedding = self.get_embedder_model(emb_type)([tag])
            
        if method == 'avg':
            tag_vecs = tag_embedding[1][0]
            tag_vec = self.vecs2vec_avg(tag_vecs) # из последовательности векторов в единый вектор (средний)
        elif method == 'tfidf':
            tag_tokens = tag_embedding[0][0]
            tag_vecs = tag_embedding[1][0]
            tag_vec = self.vecs2vec_tfidf(tag_tokens, tag_vecs, cnt)
        return tag_vec

    # Рассчет косинусного расстояния 
    # vec1 - tag (1,768), vec2 - pd.Series (df[vec_avg])
    def cos_similarity(self, vec1,vec2):
        m = vec2.shape[0] # Количество примеров в обучении
        n = vec1.shape[1] # Длина вектора 
        #vec_avg_np = vec2.to_numpy()
        vec_avg = [vec for vec in vec2]
        vec2 = np.array(vec_avg).reshape(m,n) # вектор
        scalar_product = np.dot(vec2,vec1.T)
        norm1 = np.linalg.norm(vec1) # число
        norm2 = np.linalg.norm(vec2, axis=1, keepdims=True)
        return scalar_product/(norm1*norm2)

    # Поиск релевантого класса с помощью косинусного расстояния 
    def find_relevant_name_class_cossim(self, df, tag, cnt = {}, params={}):
        method = cm.get_value_param(params, 'method', 'avg')
        threshold = cm.get_value_param(params, 'threshold', 0)
        threshold_max = cm.get_value_param(params, 'threshold_max', 0)
        num = cm.get_value_param(params, 'num', 5)
        emb_type = cm.get_value_param(params, 'emb_type', 'mult')
        find_name_for_class = cm.get_value_param(params, 'find_name_for_class', 1)
        drop_duplicates_class = cm.get_value_param(params, 'drop_duplicates_class', 1)
        
        
        # Обработка tag 
        tag = re.sub(r'[^\w\w]',' ',tag)
        tag = tag.lower()
        tag_doc = self.get_lem_model()(tag)
        tag_lem = ''
        for token in tag_doc:
            if tag_lem == '':
                tag_lem += token.lemma_
            else:
                tag_lem += ' ' + token.lemma_

        
        tag_vec = self.tag2vec(tag,method,cnt,emb_type)
        # Рассчет cos similarity между тегом и всем наимненованиями в словаре
        df['cos_similarity'] = self.cos_similarity(tag_vec,df['vec_avg'].values)
        df_sort = df.sort_values('cos_similarity',ascending=False)

        classes_list_1 = df_sort['class'].head(1).tolist()
        names_list_1 = df_sort['name'].head(1).tolist()
        first_flag = False

        if names_list_1[0] == str(classes_list_1[0]).lower():
            first_flag = True

        classes_list = []
        names_list = []
        probability_list = []
        
        if threshold == 0 and threshold_max == 0:
            classes_list = df_sort['class'].head(num).tolist()
        else:
            if threshold_max:
                classes_list = df_sort[df_sort.cos_similarity>=threshold_max]['class'].tolist()

            if not classes_list:
                classes_list = df_sort[df_sort.cos_similarity>=threshold]['class'].head(num).tolist()

            if not classes_list:
                classes_list = df_sort['class'].head(1).tolist()

        probability_list = df_sort['cos_similarity'].head(len(classes_list)).tolist()

        if find_name_for_class:
            names_list = [df[df['class']==class_]['original_name'].tolist()[0] for class_ in classes_list]
        else:
            names_list = df_sort['original_name'].head(len(classes_list)).tolist()

        for i in range(len(names_list)):
            if (names_list[i] == str(classes_list[i]).lower()) and (np.sum(df['class']==classes_list[i]) > 1):
                names_list[i] = df[(df['class']==classes_list[i]) & (df['original_name']!=names_list[i])]['original_name'].tolist()[0]

        df_temp = pd.DataFrame({'name':names_list,'class':classes_list, 'cos_similarity': probability_list})
        if drop_duplicates_class:
            df_temp = df_temp.drop_duplicates(subset=['class'])

        names_list = df_temp['name'].tolist()
        classes_list = df_temp['class'].tolist()
        probability_list = df_temp['cos_similarity'].tolist()
            
        if first_flag:
            names_list = [names_list[0]]
            classes_list = [classes_list[0]]
            probability_list = [probability_list[0]]

        return names_list, classes_list, probability_list
    

    def find_relevant_name_class_jw_from_list(self, data, tag):
        df = pd.DataFrame(data, columns=['name','class'])
        return self.find_relevant_name_class_jw(df, tag)


    # Функция отыскания кода или ID. На вход идет датафрейм c наименованием и классом, вытащенный с помощью NER тег
    def find_relevant_name_class_jw(self, df, tag):
        df['jaro_winkler_distance'] = df['name'].apply(lambda x: textdistance.jaro_winkler(x,tag))
        max_distance = df['jaro_winkler_distance'].max()
        names_list = df[df.jaro_winkler_distance==max_distance]['name'].tolist()
        classes_list = df[df.jaro_winkler_distance==max_distance]['class'].tolist()
        return names_list, classes_list 
    
    # Побочная функция перевода из array в list после применеия tag2vec, где array_ - это df['vec_avg'] после tag2vec
    def array2list(self, array_):
        list_ = array_.tolist()[0]
        return list_

    def save_csv(self, data: list, csv_path: str, pkl_path: str, params: dict = {}):
        method = cm.get_value_param(params, 'method', 'avg')
        emb_type = cm.get_value_param(params, 'emb_type', 'mult')
        # 768 координат для представления наименований в виде векторов x_1,...,x_768
        vec_name = ['x_' + str(i+1) for i in range(768)] 

        df = pd.DataFrame(data, columns=['name','class'])

        # Оставляем только уникальные наименования
        df = df.drop_duplicates(subset=['name', 'class'])

        # Создаем словарь 
        cnt = self.cnt_dict(df, save_path=pkl_path, emb_type=emb_type)

        df['name'] = df['word_emb'].apply(lambda x: ' '.join(x))
        # Получаем векторное представление для каждого имени отчета в формате np.array 
        df['vec_avg'] = df['name'].apply(lambda x: self.tag2vec(x,method,cnt=cnt,emb_type=emb_type))
        # Переводим в формат list() это представление, чтобы потом разбить на кололнки x_1,...,x_768
        df['vec_avg_list'] = df['vec_avg'].apply(lambda x: self.array2list(x))
        # Разбиваем на колонки x_1,...,x_768
        df[vec_name] = pd.DataFrame(df.vec_avg_list.tolist(), index = df.index)
        df.to_csv(csv_path, sep=';', index=False)

    # BO - '/base/.deeppavlov/downloads/cos_similarity/BO_with_vects.csv'
    # Transactions - '/base/work/train_data/reports/classification/Transaction_with_vects.csv'
    # Загружает dataframe для embedder, склеивая x_1,...,x_768 в спец array 
    def load_csv_embedder(self, load_path):
        vec_name = ['x_' + str(i+1) for i in range(768)] 
        df = pd.read_csv(load_path, sep=';')
        df['vec_avg_combined'] = df[vec_name].values.tolist()
        df['vec_avg_combined'] = df['vec_avg_combined'].apply(lambda x: np.array(x))
        df['vec_avg'] = df['vec_avg_combined']
        return df
        
    # BO - '/base/.deeppavlov/downloads/cos_similarity/dict_bo.pkl'
    # Transactions - '/base/.deeppavlov/downloads/cos_similarity/dict_transaction.pkl'    
    # Загрузка словаря для tf-idf
    def load_cnt_dict(self, load_path):
        a_file = open(load_path, "rb")
        output = pickle.load(a_file)
        return output
    
