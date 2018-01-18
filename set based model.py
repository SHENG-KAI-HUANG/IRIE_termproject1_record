# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 20:46:15 2017

@author: SHENG-KAI HUANG

注意!因為termset計算繁複，所以完整跑一次程式大概要2~5左右小時
(thresold = 1時 => 17982.422782182693 seconds.)
(thresold = 10時 => 9968.318737506866 seconds.)
(thresold = 100時 => 5763.220346689224 seconds.)
(new weight threshold = 1時 => 17972.389882326126 seconds.)
(new weight threshold = 100時 => 5756.91415143013 seconds.)
因為原本new weight是用另一個python執行
所以以上時間和這個python執行結果理論上會差一個if判斷左右的時間(位於第204行)
(new weight threshold = 10時 => 9995.543047904968 seconds.)

簡易測試的話，建議跑10~30個query(會需要大概五分鐘預處理docs.)
"""
import json;
import string;
import nltk;
import numpy as np;
from nltk.stem import WordNetLemmatizer;#使用ntlk來做lemmatization
from nltk.corpus import wordnet;#用來判斷詞性
from itertools import combinations;#用來組成termset，換句話說，index term的組合
from operator import itemgetter;
import time;

#nltk.download('wordnet');
#nltk.download('averaged_perceptron_tagger')
#假如電腦內沒有nltk相關套件的話，可能會需要這段
start = time.time();

def get_wordnet_pos(treebank_tag):#提供給Lemmatizer原本word的詞性，可以更好的找到原始字
    
    if treebank_tag.startswith('J'):
        return wordnet.ADJ;
    elif treebank_tag.startswith('V'):
        return wordnet.VERB;
    elif treebank_tag.startswith('R'):
        return wordnet.ADV;
    else:
        return wordnet.NOUN;#假如不是以上三種，則通通認為是名詞

def de_stop_word(data):
    stop_word = ['is','are','a','an','the','at','of','on','in','and'];#定義欲刪除的字
    
    for pun in string.punctuation:#先刪除除了'.'以外的所有的標點符號
        if(pun != '.'):        
            data = data.replace(pun,"");
            
    data_list= data.split();#將doc根據' '切成一個list
            
    for s in stop_word:#每次檢查一個stop word
        for i in range(0, len(data_list)):#對data_list的每個元素進行檢查，如果有元素是stop word則刪除
            if(data_list[i] == s):
                data_list[i] = '';
                
    data_final = ' '.join(data_list);#將list重新組合回string，每個list元素間差一個空白
    return data_final;

def lemmatize(data):
    WNL = WordNetLemmatizer();
    data_list2= data.split();#將doc根據' '切成一個list
    tag = nltk.pos_tag(data_list2);#找到data_list2內所有string對應的詞性

    for i in range(0, len(data_list2)):#對data_list的每個元素進行lemmatize運算
        data_list2[i] = WNL.lemmatize(data_list2[i],get_wordnet_pos(tag[i][1]));
        
    data_final2 = ' '.join(data_list2);#將list重新組合回string
    return data_final2;

def termset_function(query):#利用python內建的combinations(排列組合的組合)建立termset
    query_temp = query.split(); 
    termset_f = [];
    for i in range(1, len(query_temp) + 1):
        termset_temp = list(combinations(query_temp, i));#i代表是具有多少元素的組合
        for j in range(0, len(termset_temp)):
            termset_temp[j] = " ".join(termset_temp[j]);#因為組合的元素是list型態，所以轉回string
            termset_f.append(termset_temp[j]);#放在一組list中回傳
    return termset_f;

#f for function,e.g.,abstract_final_list_f的f就是指這是函數的參數
def create_termset_table(termset_p, abstract_final_list_f, abstract_final_set_list_f, entity_list_f):
    termset_table_final = [];
    for i in range(0, len(termset_p)):#先完成Lecture 3中Set of Termsets那一行的表示
        t= []
        t.append(termset_p[i]);
        termset_table_final.append(t);
    #所以執行完畢後termset_table_final會長得像[['asc'],['code'],...,['asc code'],...，其中'asc'為termset
    #上面for目的是為了將各termset，表示成list中的list

    for i in range(0, len(termset_p)):#對每個termset做檢查，看是否有相關聯的文件
        #這邊在切割字串一次，目的是方便查找，一次只切一個list元素
        #切完後會長得像[war], [war movie], ... => [war], [[war],[movie]] 
        termset_temp_2 = termset_p[i].split();
        
        for x in range(0,len(abstract_final_list_f)):
            number = 0;
            for j in range(0, len(termset_temp_2)):
                if abstract_final_list_f[x] is None:
                    break;
                else:
                    if termset_temp_2[j] in abstract_final_set_list_f[x]:
                        number += 1;
                    
            if number == len(termset_temp_2):#如果termset中的所有字都有出現在文件中，則紀錄文件的entity
                termset_table_final[i].append(entity_list_f[x]);
    
    return termset_table_final;

fptr_doc = open('DBdoc.json','r');
fptr_query = open('queries-v2.txt','r');
fptr_result = open('result.txt','w', encoding = 'utf-8');

json_docs = json.load(fptr_doc);#讀取dataset的json資料並放入json_docs變數中
raw_query = fptr_query.read();#讀取dataset的query資料並放入raw_query變數中
query_list = raw_query.split('\n');#先將query分成數個list元素，此時list內容為((q1_Id  query1),(q2_Id  query2)...)

threshold = int(input("請輸入threshold，只接受整數："));#可變參數
model = int(input("請輸入計算doc. weight的方式，只接受0或1，0是原版，1是長度納入考量："));
TAB_space = '	';#因為TAB'	'有時候會被縮成單空白' '，所以乾脆放在變數裡較穩定 
doc_number = 45668;
query_number = 467;

d_str_list = [];
abstract_list = [];
entity_list = [];
abstract_final_list = [];
abstract_final_set_list = [];

for i in range(0, doc_number):
     d_str_list.append(json_docs[i]);#再將list型態的str變數的元素指定給字典變數d_str
     abstract_list.append(d_str_list[i]['abstract']);#取出d_str中的abstract的內容
     if abstract_list[i] is None:
         abstract_final_list.append(abstract_list[i]);#將abstract中的stop word刪除並lemmatize運算     
         entity_list.append(d_str_list[i]['entity']);#取出d_str中的entity的內容
         abstract_final_set_list.append(abstract_list[i]);#因為是None型態，所以直接放進list中
         continue;
     else:#值得注意的是，abstract_final_list是list中有List，換句話說，[[doc1_word1,doc1_word2,...],[doc2_word1,...]]
         abstract_final_list.append(lemmatize(de_stop_word(abstract_list[i].lower())).split());#將abstract中的stop word刪除並lemmatize運算     
         entity_list.append(d_str_list[i]['entity']);#取出d_str中的entity的內容
         abstract_final_set_list.append(set(abstract_final_list[i]));#轉成set型態(以加速程式執行)後放入list中
         
for q in range(0,query_number):
    termsets = [];#存放此次query中所有termset用
    query_tf_idf = [];#存放此次query的tf-idf用
    sim_value_list = [];#存放計算完成的sim值用
    
    #將query_list的ID和query再區分開來，所以此list內容為(Id, query)
    #對query和doc做相同的lemmatize運算，希望能找到更好的結果,e.g.,(doc:run,query:running)
    single_query = (query_list[q]).split(TAB_space);
    single_query[1] = lemmatize(de_stop_word(single_query[1].lower()));#single_query[0]為Id, single_query[1]才是query
    
    #------------------------------------------------------
    #用query中的index term生成所有可能的termsets,e.g.,['vietnam','war',...,'vietnam war',...]
    termsets = termset_function(single_query[1]);
    #termset_table存放termset和其關聯的doc
    termset_table = create_termset_table(termsets, abstract_final_list,abstract_final_set_list,entity_list);
    #termset_table會長得像[[termset1,doc1_entity,doc3_entity,...],[termset2,doc2_entity,doc3_entity,...],...]
    #------------------------------------------------------
    
    #------------------------------------------------------
    #threshold發生作用，刪除關聯文件數小於threshold的termset
    delete_index = len(termsets) - 1;
    while 1 == 1:
        if delete_index <= -1:
            break;
        else:
            if (len(termset_table[delete_index]) - 1) < threshold:
                del termsets[delete_index];
                del termset_table[delete_index];
                delete_index = len(termsets) - 1;
                continue;
            else:
                delete_index -= 1;
    #------------------------------------------------------
    
    #------------------------------------------------------
    for i in range(0,len(termsets)):#計算query的tf-idf
        list_temp = [termsets[i]];
        if len(termset_table[i]) >=2:#>=2代表termset[i]有關連到的文件的時候才做計算
            temp = np.log2(1 + doc_number / (len(termset_table[i]) - 1));
        else:
            temp = 0;
        list_temp.append(temp);#list_temp此時長得像[termset[i],Wiq]
        query_tf_idf.append(list_temp);#query_tf_idf此時長得像[[termset[0],W0q],[termset[1],W1q],...]
    
    
    for i in range(0,doc_number):#計算doc的tf-idf和sim
        sim_temp = [];
        doc_tf_idf = []; 
        sim_value_temp = 0;
        if abstract_final_list[i] is not None:  
            for j in range(0,len(termsets)):#將doc-tf-idf的結果存放在doc_tf_idf串列中
                termset_temp_3 = termsets[j].split();#將termset再切成sub termset，也就是說[war movie] => [war,movie]

                Fij = abstract_final_list[i].count(termset_temp_3[0]);#計算sub termsets在文件i出現的次數
                for w in range(1, len(termset_temp_3)):#此處Fij同時代表用來找最小值
                    if Fij > abstract_final_list[i].count(termset_temp_3[w]):#找由文章每個單字組成的list中sub termset w的最小數量
                        Fij = abstract_final_list[i].count(termset_temp_3[w]);
            #為何只找最小?因為找到sub termset的最小出現次數就代表整個termset在文章中出現的次數
                if Fij != 0:
                    if model == 0:
                        temp = (1 + np.log2(Fij)) * np.log2(1 + doc_number / (len(termset_table[j]) - 1));
                    else:#如果輸入0的數字，則直接使用改變過的公式
                        temp = (1 + np.log2(Fij)) * np.log2(1 + doc_number / (len(termset_table[j]) - 1)) * len(termset_table[j][0].split());
                else:#如果有len(termset_tabel) = 1的話，那Fij也會為0，因為termset[j]和任何doc都無關聯
                    temp = 0;
                
                doc_tf_idf.append(temp);#doc_tf_idf會長得像[0, 1.2, 1.1, 0,...]，每一個元素對應到一個termset
            
            doc_long = np.sqrt(np.sum(np.square(doc_tf_idf)));#算出doc_tf_idf向量的長度，等會用在sim公式中
        
            for z in range(0,len(doc_tf_idf)):#先算出分子
                sim_value_temp += doc_tf_idf[z] * query_tf_idf[z][1];
        
            if doc_long != 0:
                sim_value = sim_value_temp / doc_long;#sim(di, q)
            else:
                sim_value = 0;
            
            if sim_value != 0:#將此份文件的sim值存放在sim_value_list中
                sim_temp.append(entity_list[i]);
                sim_temp.append(sim_value);
                sim_value_list.append(sim_temp);#等算出所有doc的分數後才會進行排序
    #------------------------------------------------------
    #不能直接對string做count，否則結果會有些錯誤，例如：'the son'.count('on') = 1 (x)
    #------------------------------------------------------
    doc_rank = sorted(sim_value_list, reverse=True,key=itemgetter(1))#等算出上一個query對所有doc的sim後，進入下一個query前，才會執行此行
    doc_rank = doc_rank[0:100];#只取前100名資料
    
    for d in range(len(doc_rank) - 1,-1,-1):#倒序輸出，也就是100,99,...,1
        str_temp = "";
        str_temp = single_query[0] + TAB_space + 'Q0' + TAB_space;
        str_temp += '<dbpedia:' + doc_rank[d][0] + '>' + TAB_space + str((d+1)) + TAB_space;
        str_temp += str( np.around(doc_rank[d][1], decimals=2)) + TAB_space + 'STANDARD' + TAB_space +'\n';
        fptr_result.write(str_temp);
    print(q + 1);#此次query的所有工作都完成後才會輸出目前完成到第幾個query
    #------------------------------------------------------
fptr_doc.close();
fptr_query.close();
fptr_result.close();

end = time.time()
elapsed = end - start
print("total Time taken: ", elapsed, "seconds.");
