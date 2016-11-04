__author__ = 'tjmy'
import os
import time
import json
import numpy as np
from collections import defaultdict
import jieba
import sys

import tornado.web
import tornado.httpserver
import tornado.ioloop
import tornado.options
from tornado.options import define, options
import tornado.httpclient

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from scipy.cluster import hierarchy

BASE_PATH = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_PATH, 'data')
STOP_PATH = os.path.join(DATA_PATH, 'stop_words')
# print(STOP_PATH)
# 停用词
stopwords = set(open(STOP_PATH, 'r', encoding='utf8').read().split())
# print(stopwords)

# define("port", default=7008, help="run on the given port", type=int)


class GetRequest(tornado.web.RequestHandler):
    def get(self, *args, **kwargs):
        result = {'code': 1, 'msg': 'this is a test', 'data': {}}
        result = json.dumps(result)
        self.write(result)

    def post(self):
        threshold = float(self.get_argument('threshold', default=-2.0))
        print('threshold:', threshold)

        infoType = int(self.get_argument('infoType', default=7))
        print('post--infoType', infoType)

        data = self.get_argument('data')

        try:
            data_list = json.loads(data)  # json格式字符串转化
            # print("received data:", data_list)
            text_cut_result = cut_text(data_list)
            # print("text_cut_result:", text_cut_result)

            if infoType == 7:  # 微博信息
                if threshold < -1.:
                    json_result = clustering(text_cut_result, threshold=0.5, infoType=infoType)
                else:
                    json_result = clustering(text_cut_result, threshold=threshold, infoType=infoType)
                self.write(json_result)
                # print('返回完毕......')
                # self.finish()
            elif infoType == 100:  # 跨媒体聚类
                if threshold < -1.:
                    json_result = clustering(text_cut_result, threshold=0.85, infoType=infoType)
                else:
                    json_result = clustering(text_cut_result, threshold=threshold, infoType=infoType)
                self.write(json_result)
            else:  # 新闻、论坛、微信等其他信息
                if threshold < -1.:
                    json_result = clustering(text_cut_result, threshold=0.6, infoType=infoType)
                else:
                    json_result = clustering(text_cut_result, threshold=threshold, infoType=infoType)
                self.write(json_result)
                # print('返回完毕......')
                # self.finish()
        except Exception:
            s = sys.exc_info()
            print("Error '%s' happened on line %d" % (s[1], s[2].tb_lineno))
            self.write(json.dumps({'code': 404}))
            # print('返回完毕......')
            # self.finish()


def cut_text(data_list):
    """
    文本预处理
    :param data_list:
    :return: text_cut_result   # 一个字典：键为id，值为一个二维list(第一维为标题words，第二维为正文words)
    """
    text_cut_result = defaultdict(list)
    try:
        for d in data_list:
            # print(d)
            precessed_data = []
            id = d['id']
            title = d['title']
            content = d['content']

            cut_title = [s for s in list(jieba.cut("".join(title.strip().split()))) if s not in stopwords]
            cut_summary = [s for s in list(jieba.cut("".join(content.strip().split()))) if s not in stopwords]
            precessed_data.append(cut_title)
            precessed_data.append(cut_summary)

            text_cut_result[id] = precessed_data
    except Exception:
        s = sys.exc_info()
        print("Error '%s' happened on line %d" % (s[1], s[2].tb_lineno))

    print("一共：", len(text_cut_result), "个案例")
    return text_cut_result


def get_center_point(cluster):
    """
    计算簇团的中心点
    :param cluster:
    :return:
    """
    center = np.zeros(len(cluster[0][1]))

    for c in cluster:
        center += c[1]
    center /= len(cluster)

    return center


def _check_hierarchy_uses_cluster_more_than_once_change(X):
    return False


def clustering(text_cut_result, threshold, infoType):
    """
    聚类
    :param text_cut_result:
    :param threshold:
    :return:
    """
    cluster_name_and_id_dict = defaultdict(list)
    try:
        ids = []
        summary_words_list = []
        title_words_list = []

        for id in text_cut_result.keys():
            ids.append(id)
            title_words_list.append(" ".join(text_cut_result[id][0]))
            summary_words_list.append(" ".join(text_cut_result[id][1]))

        s_vectorizer = CountVectorizer()
        s_transformer = TfidfTransformer()
        s_tfidf = s_transformer.fit_transform(s_vectorizer.fit_transform(summary_words_list))

        transfered_data = s_tfidf.toarray()

        # 层次聚类：根据给定threshold聚类
        # print("当前时间：", time.time())
        X = hierarchy.linkage(transfered_data, 'complete', 'cosine')
        for i in range(len(X[:, 2])):
            if X[:, 0][i] < 0:
                X[:, 0][i] = 0.0
            if X[:, 1][i] < 0:
                X[:, 1][i] = 0.0
            if X[:, 2][i] < 0:
                X[:, 2][i] = 0.0
            if X[:, 3][i] < 0:
                X[:, 3][i] = 0.0
        # print('X:', X)
        hierarchy._check_hierarchy_uses_cluster_more_than_once = _check_hierarchy_uses_cluster_more_than_once_change
        clusters = hierarchy.fcluster(Z=X, t=threshold, criterion='distance')

        # print(clusters)
        for k in range(min(clusters), max(clusters) + 1):
            # print("cluster_name:", k)
            cluster_points = []
            for i in range(len(transfered_data)):
                if clusters[i] == k:
                    cluster_points.append(ids[i])
            # print("该簇团元素个数：", len(cluster_points))
            if len(cluster_points) == 0:
                continue
            cluster_name_and_id_dict[cluster_points[0]] = cluster_points
        # print('一次聚类完成......')

        if int(infoType) != 7:  # 对于新闻信息，需要根据标题进行二次聚类
            id1_list = []
            id2_list = []
            for key in cluster_name_and_id_dict.keys():
                id1_list.append(key)
                id2_list.append(key)

            removed_list = []
            for id1 in id1_list:
                if len(cluster_name_and_id_dict[id1]) == 1:
                    for id2 in id2_list:
                        if id1 == id2 and id2 in removed_list:
                            continue
                        two_title_words_list = [title_words_list[ids.index(id1)], title_words_list[ids.index(id2)]]

                        s_tfidf = s_transformer.fit_transform(s_vectorizer.fit_transform(two_title_words_list))

                        trans_data = s_tfidf.toarray()
                        sim = cosine_similarity(trans_data[0], trans_data[1])
                        # print(sim)
                        if sim is not None and sim > 0.7:
                            cluster_name_and_id_dict[id2].append(id1)  # 合并
                            del cluster_name_and_id_dict[id1]  # 删除被合并的元素
                            removed_list.append(id1)
                            break

        # print('二次聚类完成......')
        del s_vectorizer
        del s_transformer
        print('cluster finished ^_^')
    except Exception:
        s = sys.exc_info()
        print("Error '%s' happened on line %d" % (s[1], s[2].tb_lineno))

    count = 0
    for key in cluster_name_and_id_dict.keys():
        count += len(cluster_name_and_id_dict[key])
        # print(key, ":", cluster_name_and_id_dict[key])
    print('after clustering finished, the points are:', count)
    return json.dumps(cluster_name_and_id_dict)


def cosine_similarity(vector1, vector2):
    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    for a, b in zip(vector1, vector2):
        dot_product += a * b
        normA += a ** 2
        normB += b ** 2
    if normA == 0.0 or normB == 0.0:
        return None
    else:
        return dot_product / ((normA * normB) ** 0.5)


def make_app():
    return tornado.web.Application(handlers=[
        (r"/", GetRequest)
    ])


if __name__ == '__main__':
    # tornado.options.parse_command_line()
    # app = make_app()
    # http_server = tornado.httpserver.HTTPServer(app)
    # http_server.listen(options.port)
    # tornado.ioloop.IOLoop.instance().start()

    app = make_app()
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.bind(7018)
    http_server.start(5)  # Forks multiple sub-processes
    tornado.ioloop.IOLoop.instance().start()
