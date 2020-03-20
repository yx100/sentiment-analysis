import argparse
import pickle as pkl
import numpy as np
from tqdm import tqdm
from time import gmtime, strftime
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from sklearn.metrics import classification_report
from models import EmojiSentimentClassification

from Data_weibo import TuplesListDataset, Vectorizer, BucketSampler, emotion_dict,get_inverse_dict
import random

torch.cuda.set_device(0)
inverse_dict = None
inverse_emoji_dic = None
def checkpoint(epoch,model,output):
    model_out_path = output+"_epoch_{}.pth".format(epoch)
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

def check_memory(emb_size,max_sents,max_words,b_size,cuda):
    try:
        e_size = (2,b_size,max_sents,max_words,emb_size) 
        d_size = (b_size,max_sents,max_words)
        t = torch.rand(*e_size)
        db = torch.rand(*d_size)

        if cuda:
            db = db.cuda()
            t = t.cuda()
        print("-> Quick memory check : OK\n")

    except Exception as e:
        print(e)
        print("Not enough memory to handle current settings {} ".format(e_size))
        print("Try lowering sentence size and length.")
        sys.exit()

def load_embeddings(file):
    emb_file = open(file).readlines()
    first = emb_file[0]
    word, vec = int(first.split()[0]),int(first.split()[1])
    size = (word,vec)
    print("--> Got {} words of {} dimensions".format(size[0],size[1]))
    tensor = np.zeros((size[0]+2,size[1]),dtype=np.float32) ## adding padding + unknown
    word_d = {}
    word_d["_padding_"] = 0
    word_d["_unk_word_"] = 1

    print("--> Shape with padding and unk_token:")
    print(tensor.shape)

    for i,line in tqdm(enumerate(emb_file,1),desc="Creating embedding tensor",total=len(emb_file)):
        if i==1: 
            continue

        spl = line.strip().split(" ")

        if len(spl[1:]) == size[1]: 
            word_d[spl[0]] = i
            tensor[i] = np.array(spl[1:],dtype=np.float32)
        else:
            print("WARNING: MALFORMED EMBEDDING DICTIONNARY:\n {} \n line isn't parsed correctly".format(line))

    try:
        assert(len(word_d)==size[0]+2)
    except:
        print("Final dictionnary length differs from number of embeddings - some lines were malformed.")
    return tensor, word_d

def load_emoji_embeddings(file):
    lines = open(file).readlines()
    line_0 = lines[0].strip().split(" ")
    dim= len(line_0[1:])

    emoji_word={}
    emoji_word["_unk_word_"] = 0
    emoji_tensor = np.zeros((len(lines) + 1, dim), dtype=np.float32) ## adding padding + unknown
    for i, line in enumerate(lines, 1):
        emoji_vector = line.strip().split(" ")
        emoji_word[emoji_vector[0]] = i
        emoji_tensor[i] = np.array(emoji_vector[1:], dtype=np.float32)

    return emoji_tensor, emoji_word

def  reset_mapping(emoji_mapping, emoji_dic):
    emoji_map= {}
    emoji_map["_unk_word_"] = 0
    emoji_map["_padding_"] = 1
    for k,v in emoji_mapping.items():
        for dic_k, dic_v in emoji_dic.items():
            if  k==dic_k:
                idx = dic_v
                emoji_map[len(emoji_map)]=idx

    return emoji_map

def save(model,dic,path):
    dict_m = model.state_dict()
    dict_m["word_dic"] = dic
    dict_m["reviews"] = torch.Tensor()
    dict_m["word.mask"] = torch.Tensor()
    dict_m["sent.mask"] = torch.Tensor()

    torch.save(dict_m,path)

def tuple_batcher_builder(vectorizer, trim=True):
    def tuple_batch(l):
        emoji, item, review, rating, emotion_category = zip(*l)
        list_rev = vectorizer.vectorize_batch(review, trim)       
        
        stat = sorted([(len(s), len(r), r_n, s_n, s) for r_n, r in enumerate(list_rev) for s_n, s in enumerate(r)], reverse=True)

        emoji = [emoji[r_n] for ls, lr, r_n, s_n, _ in stat]
        item = [item[r_n] for ls, lr, r_n, s_n, _ in stat]
        review = [review[r_n] for ls, lr, r_n, s_n, _ in stat]
        rating = [rating[r_n] for ls, lr, r_n, s_n, _ in stat]
        emotion_category = [emotion_category for ls, lr, r_n, s_n, _ in stat]
        r_t = torch.Tensor(rating).long()
        u_t = torch.Tensor(emoji).long()
        i_t = torch.Tensor(item).long()

        max_len = stat[0][0]
        batch_t = torch.zeros(len(stat), max_len).long()

        for i, s in enumerate(stat):
            for j, w in enumerate(s[-1]):  
                batch_t[i, j] = w

        stat = [(ls, lr, r_n, s_n) for ls, lr, r_n, s_n, _ in stat]

        return batch_t, r_t, u_t, i_t, stat, review

    return tuple_batch


def tuple2var(tensors,data):
    def copy2tensor(t,data):
        t.resize_(data.size()).copy_(data)
        return Variable(t)
    return tuple(map(copy2tensor,tensors,data))

def new_tensors(n,cuda,types={}):
    def new_tensor(t_type,cuda):
        x = torch.Tensor()
        if t_type:
            x = x.type(t_type)
        if cuda:
            x = x.cuda()
        return x

    return tuple([new_tensor(types.setdefault(i,None),cuda) for i in range(0,n)])

def train(epoch,model,optimizer,dataset,criterion,cuda):
    epoch_loss = 0
    ok_all = 0
    ok_all = 0
    data_tensors = new_tensors(4,cuda,types={0:torch.LongTensor,1:torch.LongTensor,2:torch.LongTensor,3:torch.LongTensor}) #data-tensors

    with tqdm(total=len(dataset),desc="Training") as pbar:
        for iteration, (batch_t,r_t, u_t, i_t,stat,rev) in enumerate(dataset):
            data = tuple2var(data_tensors,(batch_t,r_t,u_t,i_t))

            optimizer.zero_grad()
            out = model(data[0],data[2],data[3],stat,rev)

            ok,per = accuracy(out,data[1])
            loss = criterion(out, data[1])
            epoch_loss += loss.data[0]
            loss.backward()
            optimizer.step()

            ok_all += per.data[0]

            pbar.update(1)
            pbar.set_postfix({"acc":ok_all/(iteration+1),"CE":epoch_loss/(iteration+1)})

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}, {}% accuracy".format(epoch, epoch_loss /len(dataset),ok_all/len(dataset)))

def test_by_emotion(epoch,model,dataset,cuda,msg="test_by_emotion Evaluating"):
    epoch_loss = 0
    ok_all = 0
    pred = 0
    skipped = 0
    data_tensors = new_tensors(4,cuda,types={0:torch.LongTensor,1:torch.LongTensor,2:torch.LongTensor,3:torch.LongTensor}) #data-tensors
    all_truch = []
    all_predict = []
    with tqdm(total=len(dataset),desc=msg) as pbar:
        for iteration, (batch_t,r_t,u_t,i_t, stat,rev) in enumerate(dataset):
            data = tuple2var(data_tensors,(batch_t,r_t,u_t,i_t))
            out = model(data[0],data[2],data[3],stat)
            ok,per = accuracy(out,data[1])
            ok_all += per.data[0]
            pred+=1
            all_predict.append(argmax(out))
            all_truch.append(data[1].clone())
            pbar.update(1)
            pbar.set_postfix({"acc":ok_all/pred, "skipped":skipped})

    all_predict_ = torch.cat(all_predict)
    all_truch_ = torch.cat(all_truch)
    print("===> {} Complete:  {}% accuracy".format(msg,ok_all/pred))


def test(i, epoch,net,dataset,cuda,msg="Evaluating"):
    epoch_loss = 0
    ok_all = 0
    pred = 0
    skipped = 0
    data_tensors = new_tensors(4,cuda,types={0:torch.LongTensor,1:torch.LongTensor,2:torch.LongTensor,3:torch.LongTensor}) #data-tensors
    all_emoji = []
    all_truch = []
    all_predict = []
    all_revs = []

    with tqdm(total=len(dataset),desc=msg) as pbar:
        for iteration, (batch_t,r_t,u_t,i_t, stat,rev) in enumerate(dataset):
            data = tuple2var(data_tensors,(batch_t,r_t,u_t,i_t))
            out = net(data[0],data[2],data[3],stat,rev)

            ok,per = accuracy(out,data[1])
            ok_all += per.data[0]
            pred+=1

            all_predict.append(argmax(out))
            all_truch.append(data[1].clone())
            all_emoji.append(u_t.clone())
            all_revs += list(rev)
            pbar.update(1)
            pbar.set_postfix({"acc":ok_all/pred, "skipped":skipped})

    if msg == "Evaluating":
        all_predict_ = torch.cat(all_predict)
        all_truch_ = torch.cat(all_truch)
        all_predict_ = all_predict_.data.cpu().numpy()
        all_truch_ = all_truch_.data.cpu().numpy()
        all_emoji = torch.cat(all_emoji).cpu().numpy()
        global inverse_emoji_dic
        all_emoji_ = [inverse_emoji_dic[x] for x in all_emoji]
        test_result = zip(all_predict_,all_truch_,all_revs, all_emoji_)
        accu = str(ok_all/pred)
        finename_base1 = "/home/lyx/py-workspace/lstm-weibo-1002/output/attention/total/tt" + str(i)+ "_"+ str(epoch) +"_"+accu
        current_time_file1 = strftime(finename_base1, gmtime())
       
        with  open(current_time_file1, 'w') as f1:
            for (pred_po, truth_po, rev, emoji) in test_result:
      
                line = "{}\t{}\t{}\t{}\n".format(pred_po, truth_po, rev, emoji)
                f1.write(line)
        
        print(classification_report(all_truch_, all_predict_,digits=4))

    print("===> {} Complete:  {}% accuracy".format(msg,ok_all/pred))

def argmax(out):
    def sm(mat):
        exp = torch.exp(mat)
        sum_exp = exp.sum(1,True)+0.0001
        return exp/sum_exp.expand_as(exp)

    _,max_i = torch.max(sm(out),1)
    return max_i

def accuracy(out,truth):
    def sm(mat):
        exp = torch.exp(mat)
        sum_exp = exp.sum(1,True)+0.0001
        return exp/sum_exp.expand_as(exp)

    _,max_i = torch.max(sm(out),1)
   
    eq = torch.eq(max_i,truth).float()
    all_eq = torch.sum(eq)

    return all_eq, all_eq/truth.size(0)*100

def shuffle(tuples):
    random.seed(4000)
    random.shuffle(tuples)

def expand(data_tuples):
    '''
    :param (emotion, emotion, review, truth_category)
    :return: (emotion, emotion, review, truth_category, predicted_only_by_emotion)
    '''
    result = []
    for t in data_tuples:
        emoji = t[0]
        if emoji in emotion_dict.keys():
            emoji_category = emotion_dict.get(emoji, 1)
        else:
            print("emoji not in dict", emoji)

        emoji_category = int(emoji_category)
        result.append(t + (emoji_category, ))
    return result

def accuracy_by_emoji(data_tuples):
    '''
    calculate the accuracy of emotion category
    :param data_tuples:
    :return: accuracy
    '''
    truth = [x[3] for x in data_tuples]
    predict = [x[4] for x in data_tuples]
    import numpy as np
    truth = np.asarray(truth, dtype=np.int32)
    predict = np.asarray(predict, dtype=np.int32)
    return (truth == predict).sum() / truth.size

def main(args):
    print(32*"-"+"\EmojiSentimentClassification Attention Network:\n" + 32*"-")
    print("\nLoading Data:\n" + 25*"-")

    max_features = args.max_feat
    datadict = pkl.load(open(args.filename,"rb"))
    tuples = datadict["data"]
    tuples = expand(tuples)
    acc_by_emoji =  accuracy_by_emoji(tuples)
    print("accuracy by emoji: {}".format(acc_by_emoji))
    splits  = datadict["splits"]
    split_keys = set(x for x in splits)
    shuffle(tuples)
    if args.split not in split_keys:
        print("Chosen split (#{}) not in split set {}".format(args.split,split_keys))
    else:
        print("Split #{} chosen".format(args.split))

    train_sets,test_sets = TuplesListDataset.build_train_test(tuples,splits)

    tensor, dic = load_embeddings(args.emb)
    global inverse_dict
    inverse_dict = get_inverse_dict(dic)
    print(len(dic))

    emoji_tensor, emoji_dic = load_emoji_embeddings(args.emoji )
    global inverse_emoji_dic
    inverse_emoji_dic = get_inverse_dict(emoji_dic)


    for i,(train_set, test_set) in enumerate(zip(train_sets, test_sets)):
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print("training process: {}".format(i))
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

        emoji_mapping = train_set.set_mapping(0,  offset=1, dict= emoji_dic )  
        item_mapping = train_set.set_mapping(1,  offset=1, dict= emoji_dic)
        
        classes = train_set.set_mapping(3) 
        test_set.set_mapping(3,dict = classes) 
        test_set.set_mapping(0,dict=emoji_mapping) 
        test_set.set_mapping(1,dict=emoji_mapping) 
        num_class = len(classes)

        print(25*"-"+"\nClass stats:\n" + 25*"-")
        print("Train set:\n" + 10*"-")

        class_stats,class_per = train_set.get_stats(3)
        print(class_stats)
        print(class_per)

        if args.weight_classes:
            class_weight = torch.zeros(num_class)
            for c,p in class_per.items():
                class_weight[c] = 1-p
            print(class_weight)

            if args.cuda:
                class_weight = class_weight.cuda()

        print(10*"-" + "\n Test set:\n" + 10*"-")

        test_stats,test_per = test_set.get_stats(3)
        print(test_stats)
        print(test_per)

        print(25*"-" + "\nBuilding word vectors: \n"+"-"*25)
        vectorizer = Vectorizer(max_word_len=args.max_words,max_sent_len=args.max_sents)

        if args.load:
            state = torch.load(args.load)
            vectorizer.word_dict = state["word_dic"]
            net = EmojiSentimentClassification(ntoken=len(state["word_dic"]), nemojis=len(emoji_tensor), nitems=len(emoji_tensor) ,
                                         emb_size=state["embed.weight"].size(1),hid_size=state["sent.gru.weight_hh_l0"].size(1),
                                         num_class=state["lin_out.weight"].size(0))

            net.load_state_dict(state)
        else:
            if args.emb:

                net = EmojiSentimentClassification(ntoken=len(dic), nemojis=len(emoji_tensor), nitems=len(emoji_tensor), num_class=num_class,emb_size=len(tensor[1]),
                                      hid_size=args.hid_size)

                net.set_emb_tensor(torch.FloatTensor(tensor), torch.FloatTensor(emoji_tensor), zero_weight=False,auto_weight=False)


                vectorizer.word_dict = dic
            else:
                vectorizer.build_dict(train_set.field_gen(2),args.max_feat)
                net = HierarchicalDoc(ntoken=len(vectorizer.word_dict),nemojis=len(emoji_tensor), nitems=len(emoji_tensor) , emb_size=args.emb_size,hid_size=args.hid_size, num_class=num_class)

        tuple_batch = tuple_batcher_builder(vectorizer,trim=True)
        tuple_batch_test = tuple_batcher_builder(vectorizer,trim=True)

        if args.balance:
            pass

        else:
            dataloader = DataLoader(train_set, batch_size=args.b_size, shuffle=True, num_workers=2, collate_fn=tuple_batch,pin_memory=True)
            dataloader_valid = DataLoader(val_set, batch_size=args.b_size, shuffle=False,  num_workers=2, collate_fn=tuple_batch_test)
            dataloader_test = DataLoader(test_set, batch_size=args.b_size, shuffle=False, num_workers=2, collate_fn=tuple_batch_test)

        if args.weight_classes:
            criterion = torch.nn.CrossEntropyLoss(weight=class_weight)
        else:
            criterion = torch.nn.CrossEntropyLoss()
        if args.cuda:
            net.cuda()

        print("-"*20)
        check_memory(args.max_sents,args.max_words,net.emb_size,args.b_size,args.cuda)

        optimizer = optim.Adam(net.parameters())
        torch.nn.utils.clip_grad_norm(net.parameters(), args.clip_grad)


        for epoch in range(1, args.epochs + 1):
            train(epoch,net,optimizer,dataloader,criterion,args.cuda)

            test(i, epoch,net,dataloader_test,args.cuda)
            current_time_file = "checkpoint/"+strftime("%Y-%m-%d-%H:%M:%S", gmtime())
            checkpoint(epoch,net,current_time_file)


        if args.save:
            print("model saved to {}".format(args.save))
            save(net,vectorizer.word_dict,args.save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Emoji Attention Networks for Sentiment analysis')
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--emb-size",type=int,default=200)
    parser.add_argument("--hid-size",type=int,default=50)
    parser.add_argument("--weight-classes", action='store_true')
    parser.add_argument("--b-size", type=int, default=16)
    parser.add_argument("--max-feat", type=int,default=10000)
    parser.add_argument("--epochs", type=int,default=30)
    parser.add_argument("--clip-grad", type=float,default=1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--max-words", type=int,default=32)
    parser.add_argument("--max-sents", type=int, default=16)
    parser.add_argument("--momentum",type=float,default=0.9)
    parser.add_argument("--emb", type=str)
    parser.add_argument("--load", type=str)
    parser.add_argument("--save", type=str)
    parser.add_argument("--snapshot", action='store_true')
    parser.add_argument("--output", type=str)
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument('--balance', action='store_true',  help='balance class in batches')
    parser.add_argument('filename', type=str)
    parser.add_argument('--emoji', type=str)
    args = parser.parse_args()

    main(args)
