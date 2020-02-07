import numpy as np
import sklearn
import xlrd
import jieba
import datetime
import re
import opencc
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix,accuracy_score, precision_score, recall_score,f1_score
from sklearn import naive_bayes
from sklearn import tree
from gensim.models import FastText
from gensim.models.word2vec import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from imblearn.over_sampling import SMOTE
import sys
sys.setrecursionlimit(3000)
t = datetime.datetime
#=============================================================================================================
def patents_to_weighted_vec(x,tx,model,tfidf_vec, tfidf_matrix):
    tfidf_words = tfidf_vec.get_feature_names() # TF-IDF 所有字的列表
    tfidf_dict  = tfidf_vec.vocabulary_ # TF-IDF 每個字對應的 ID
    tfidf_weight = tfidf_matrix.toarray() # 將向量轉成陣列方便使用（容易當機）
    
    xVec = []
    txVec = []
    for doc in x:
        # 確認兩邊都存在的詞
        doc = [word for word in doc if word in model.wv.vocab and word in tfidf_words]
        # 將 TF-IDF 乘上 Word2Vec
        doc = [tfidf_weight[tfidf_dict[word]].T * model.wv[word] for word in doc]
        # 平均所有字
        vec = np.mean(np.array(doc), axis=0)
        xVec.append(vec)
    for doc in tx:
        # 確認兩邊都存在的詞
        doc = [word for word in doc if word in model.wv.vocab and word in tfidf_words]
        # 將 TF-IDF 乘上 Word2Vec
        doc = [tfidf_weight[tfidf_dict[word]].T * model.wv[word] for word in doc]
        # 平均所有字
        vec = np.mean(np.array(doc), axis=0)
        txVec.append(vec)
    return xVec , txVec
#=============================================================================================================
def vectorizer(model,x,tx):
    xVec = []
    txVec = []
    for i in range(len(x)):
        temp = []
        for j in range(len(x[i])):
            if(x[i][j] in model.wv):
                temp.append(model.wv[x[i][j]])
        xVec.append(sum(temp) / len(temp))
    for i in range(len(tx)):
        temp = []
        for j in range(len(tx[i])):
            if(tx[i][j] in model.wv):
                temp.append(model.wv[tx[i][j]])
        txVec.append(sum(temp) / len(temp))
    return xVec , txVec
#=============================================================================================================
def nnclass(calss,xVec,txVec,y,ty):
    nnErr = 0.0
    nnlist = []
    for k in range(len(txVec)):
        dist = []
        for m in range(len(xVec)):
            dist.append(np.sum(np.abs(txVec[k] - xVec[m])))
        nnlist.append(y[dist.index(min(dist))])
        if(y[dist.index(min(dist))] != ty[k]):
            nnErr += 1
    nnlist = np.asarray(nnlist)
    matrix(calss + '_NN',ty,nnlist)
    return 100-( nnErr / len(ty) * 100)
#=============================================================================================================
def randomforestcalss(calss,xVec,txVec,y,ty):
    rfErr = 0.0
    
    clf = RandomForestClassifier(n_estimators=50, max_depth=100, random_state = 0)
    clf.fit(xVec,y)
    result = clf.predict(txVec)
    for k in range(len(result)):
        if(result[k] != ty[k]):
            rfErr +=1
    matrix(calss + '_RF',ty,result)
    return 100-( rfErr / len(ty) * 100)
#=============================================================================================================
def svmclass(calss,xVec,txVec,y,ty):
    svmErr = 0.0
    
    clf = sklearn.svm.SVC(C=0.1,gamma='auto')
    clf.fit(xVec,y)
    result = clf.predict(txVec)
    for k in range(len(result)):
        if(result[k] != ty[k]):
            svmErr +=1
    matrix(calss + '_SVM',ty,result)
    return 100-( svmErr / len(ty) * 100)
#=============================================================================================================
def gaussianNBclass(calss,xVec,txVec,y,ty):
    gnbErr = 0.0

    gnb = naive_bayes.GaussianNB()
    gnb.fit(xVec,y)
    result = gnb.predict(txVec)
    for k in range(len(result)):
        if(result[k] != ty[k]):
            gnbErr +=1
    matrix(calss + '_gaussianNB',ty,result)
    return 100-( gnbErr / len(ty) * 100)
#=============================================================================================================
def Decisiontreeclass(calss,xVec,txVec,y,ty):
    dctErr = 0.0
    clf = tree.DecisionTreeClassifier()
    clf.fit(xVec,y)
    result = clf.predict(txVec)
    for k in range(len(result)):
        if(result[k] != ty[k]):
            dctErr +=1
    matrix(calss + '_Decisiontree',ty,result)
    return 100-( dctErr / len(ty) * 100)
#=============================================================================================================
def xgbclass(calss,xVec,txVec,y,ty,gg,kkk):
    dtrain = xgb.DMatrix(xVec, label=y)
    dtest = xgb.DMatrix(txVec, label=ty)
     # label必須從 0 開始
    param = {'max_depth':6,'gamma':0.2, 'eta':0.3, 'objective': 'multi:softmax', 'eval_metric':'merror', 'silent':1,'num_class':4} 
    evallist  = [(dtrain,'train'), (dtest,'test')]
    num_round = 75  # 循環次數
    bst = xgb.train(param, dtrain, num_round, evallist)
    preds = bst.predict(dtest)
    
    predictions = [round(value) for value in preds] #訓練出來的類別
    y_test = dtest.get_label()
    test_accuracy = sklearn.metrics.accuracy_score(y_test, predictions)
    matrix(calss + '_XGBoost',y_test,predictions)
    print("[%s]第 %d 次中第 %d 次的訓練" % (t.now() , gg , kkk))
    return test_accuracy * 100.0
#=============================================================================================================
def process_and_run(DocumentList , Labellist , gg , tfidf_vec, tfidf_matrix):
    DocumentList = np.array(DocumentList)
    Labellist = np.array(Labellist)
    skf = StratifiedKFold(n_splits = 10 , shuffle = True)
    kkk = 0
    for x_index , tx_index in skf.split(DocumentList , Labellist):
        x_temp = []
        for x_data in DocumentList[x_index]:
            x_temp.append(x_data)
        tx_temp = []
        for tx_data in DocumentList[tx_index]:
            tx_temp.append(tx_data)
        x = x_temp
        tx = tx_temp
        y = Labellist[x_index]
        ty = Labellist[tx_index]
        kkk += 1
        print("[{}]訓練model中...".format(t.now()))
        model_w2v,model_FT = load_model(x)
        #建立向量
        tfidf_x , tfidf_tx = patents_to_weighted_vec(x,tx,model_w2v,tfidf_vec,tfidf_matrix)
        tfidf_xVec ,tfidf_txVec, tfidf_t, tfidf_ty = bigdata(tfidf_x , tfidf_tx,y,ty)
        #TFIDF*word2vec向量模型
        tfidf_xgb.append(xgbclass('tfidf',tfidf_xVec ,tfidf_txVec, tfidf_t, tfidf_ty,gg,kkk))
        tfidf_NN.append(nnclass('tfidf',tfidf_xVec ,tfidf_txVec, tfidf_t, tfidf_ty))
        tfidf_RF.append(randomforestcalss('tfidf',tfidf_xVec ,tfidf_txVec, tfidf_t, tfidf_ty))
        tfidf_SVM.append(svmclass('tfidf',tfidf_xVec ,tfidf_txVec, tfidf_t, tfidf_ty))
        tfidf_GB.append(gaussianNBclass('tfidf',tfidf_xVec ,tfidf_txVec, tfidf_t, tfidf_ty))
        tfidf_DT.append(Decisiontreeclass('tfidf',tfidf_xVec ,tfidf_txVec, tfidf_t, tfidf_ty))
        #建立向量
        w2c_x , w2c_tx = vectorizer(model_w2v,x,tx)
        w2c_xVec ,w2c_txVec , w2c_y ,w2c_ty = bigdata(w2c_x , w2c_tx,y,ty)
        #word2vec向量模型
        w2v_xgb.append(xgbclass('word2vec',w2c_xVec ,w2c_txVec , w2c_y ,w2c_ty,gg,kkk))
        w2v_NN.append(nnclass('word2vec',w2c_xVec ,w2c_txVec , w2c_y ,w2c_ty))
        w2v_RF.append(randomforestcalss('word2vec',w2c_xVec ,w2c_txVec , w2c_y ,w2c_ty))
        w2v_SVM.append(svmclass('word2vec',w2c_xVec ,w2c_txVec , w2c_y ,w2c_ty))
        w2v_GB.append(gaussianNBclass('word2vec',w2c_xVec ,w2c_txVec , w2c_y ,w2c_ty))
        w2v_DT.append(Decisiontreeclass('word2vec',w2c_xVec ,w2c_txVec , w2c_y ,w2c_ty))
        #建立向量
        ft_x , ft_tx = vectorizer(model_FT,x,tx)
        ft_xVec,ft_txVec,ft_y,ft_ty = bigdata(ft_x , ft_tx,y,ty)
        #fasttext向量模型
        ft_xgb.append(xgbclass('fasttext',ft_xVec,ft_txVec,ft_y,ft_ty,gg,kkk))
        ft_NN.append(nnclass('fasttext',ft_xVec,ft_txVec,ft_y,ft_ty))
        ft_RF.append(randomforestcalss('fasttext',ft_xVec,ft_txVec,ft_y,ft_ty))
        ft_SVM.append(svmclass('fasttext',ft_xVec,ft_txVec,ft_y,ft_ty))
        ft_GB.append(gaussianNBclass('fasttext',ft_xVec,ft_txVec,ft_y,ft_ty))
        ft_DT.append(Decisiontreeclass('fasttext',ft_xVec,ft_txVec,ft_y,ft_ty))
#=============================================================================================================
def bigdata(x,tx,y,ty):
    x_smo, y_smo = smo.fit_sample(x, y)
    tx_smo, ty_smo = smo.fit_sample(tx, ty)
    return x_smo ,tx_smo, y_smo, ty_smo
#=============================================================================================================
def load_model(DocumentList):
    model_w2v = Word2Vec(DocumentList,
                         size = 300,
                         iter = 10,
                         sg = 1,
                         workers = 11,
                         negative = 5,
                         min_count = 5,
                         max_vocab_size = None)
    model_FT = FastText(DocumentList,
                        size = 300,
                        iter = 10,
                        min_count = 5,
                        max_vocab_size=None)
    return model_w2v,model_FT
#=============================================================================================================
def train_tfidf(DocumentList):
    print('[%s] 開始訓練 TF-IDF' %t.now())
    tfidf_vec = TfidfVectorizer(max_features = 300,
                                min_df = 5,
                                analyzer = 'word',
                                preprocessor = fake_tokenizer,
                                tokenizer = fake_tokenizer,
                                lowercase = False)
    tfidf_matrix = tfidf_vec.fit_transform(DocumentList) # 訓練 TF-IDF
    
    return tfidf_vec, tfidf_matrix
#=============================================================================================================
def fake_tokenizer(DocumentList):
    '''
    假的斷詞器，用來跳過 TF-IDF 的斷詞步驟
    '''
    return DocumentList
#=============================================================================================================
def matrix(name,ty , result):
    #precision,recall,accuracy,f1的結果
    confusion_matrix(ty, result)
    #precision = tp/(tp+fp)
    precision = precision_score(ty, result,average=None)
    #recall = tp/(tp+fn)
    recall = recall_score(ty, result,average=None)
    # accuracy = (TP + TN) / (TP + FN + FP + TN)
    accuracy = accuracy_score(ty, result)
    #f1 = 2*precision*recall/ (precision+recall)
    f1 = f1_score(ty, result,average=None)
    Precision = precision.mean()
    Recall = recall.mean()
    Accuracy = accuracy.mean()
    F1 = f1.mean()
    with open('log/matrix/'+ name + '.csv' , 'a') as f:
        f.write(str(Precision) + "," + str(Recall) + "," + str(Accuracy) + "," + str(F1) + "\n")

#=============================================================================================================
def stopword():
    stopwords = []
    with open('dict/stop_word.txt','r') as f:
        for line in f:
            for word in line.split('\n'):
                word = cc.convert(word)
                stopwords.append(word)
    return stopwords
#=============================================================================================================
def load_data():
    # 打開訓練資料
    book = xlrd.open_workbook(xls)
    stop_words = stopword()
    # 打開指定的分頁
    sheet = book.sheet_by_index(0)
    for row_index in range(0, sheet.nrows):
        # 這邊把文字內容先做個簡單的整理
        content = sheet.cell(row_index, 2).value # 內容
        quality = int(sheet.cell(row_index, 0).value) - 2  # label
        content_no_digit = ''.join([i for i in content if not i.isdigit()])  # 去除數字
        news = re.sub('\W', '', content_no_digit)  # 正規化 去除標點符號
        news = cc.convert(news) # 轉成簡體
        jiebaBox = jieba.cut(news, cut_all=False)
        jiebaBox = [t for t in jiebaBox if t not in stop_words]
        temp = []
        #加變數
        for t in jiebaBox:
            temp.append(t)
        DocumentList.append(temp) # 放到文件列表裡
        Labellist.append(quality) # 放到label列表裡
    return DocumentList , Labellist
#=============================================================================================================
def print_classification_result(classname,method):
    i = 0
    avg = 0
    for fksc in method:
        #print("W2V round %s : %s" %(str(i), str(fksc)))#印每一次
        i += 1
        avg += fksc
    with open('log/log.txt','a') as f:
        f.write("[%s]%s %s次的平均: %s \n" % (t.now(),classname,str(i/10),str(avg/(i))))
#=============================================================================================================
xls = '新聞all.xlsx'
cc = opencc.OpenCC('t2s')
smo = SMOTE(random_state=42)
DocumentList = []
Labellist = []
tfidf_xgb = []
tfidf_NN = []
tfidf_RF = []
tfidf_SVM = []
tfidf_GB = []
tfidf_DT = []
w2v_xgb = []
w2v_NN = []
w2v_RF = []
w2v_SVM = []
w2v_GB = []
w2v_DT = []
ft_xgb = []
ft_NN = []
ft_RF = []
ft_SVM = []
ft_GB = []
ft_DT = []
#=============================================================================================================
def main():
    print("[{}]讀取資料中...".format(t.now()))
    jieba.load_userdict('dict/dict3.txt')
    DocumentList , Labellist = load_data()
    tfidf_vec, tfidf_matrix = train_tfidf(DocumentList)
    #print("[{}]存入斷詞結果...".format(t.now()))
    #with open('jieba_word/jieba_words.txt', 'w',encoding='utf-8') as f:
    #    f.write(str(DocumentList))
    for gg in range(100):
        process_and_run(DocumentList , Labellist , gg+1, tfidf_vec, tfidf_matrix)
    #process_and_run(DocumentList , Labellist, 1, tfidf_vec, tfidf_matrix)
    allclass = [tfidf_xgb,tfidf_NN,tfidf_RF,tfidf_SVM,tfidf_GB,tfidf_DT,
                w2v_xgb,w2v_NN,w2v_RF,w2v_SVM,w2v_GB,w2v_DT,
                ft_xgb,ft_NN,ft_RF,ft_SVM,ft_GB,ft_DT]
    allclass_name = ['tfidf_xgb','tfidf_NN','tfidf_RF','tfidf_SVM','tfidf_GB','tfidf_DT',
                     'w2v_xgb','w2v_NN','w2v_RF','w2v_SVM','w2v_GB','w2v_DT',
                     'ft_xgb','ft_NN','ft_RF','ft_SVM','ft_GB','ft_DT']
    i = 0
    for gbl in allclass:
        print_classification_result(allclass_name[i],gbl)
        i+= 1
#=============================================================================================================
if __name__ == "__main__":
    main()