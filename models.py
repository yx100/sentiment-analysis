import torch
import torch.nn as nn
import torch.nn.functional as F
from operator import itemgetter
from torch.autograd import Variable
from collections import OrderedDict

global weight_dict
weight_dict={}

class AttentionalBiGRU(nn.Module):

    def __init__(self, inp_size, hid_size, dropout=0.3,  no_att=False):
        super(AttentionalBiGRU, self).__init__()
        self.register_buffer("mask",torch.FloatTensor())

        natt = hid_size*2
        self.gru = nn.LSTM(input_size=inp_size,hidden_size=hid_size,num_layers=1,bias=True,batch_first=True,dropout=dropout,bidirectional=True)
     
        self.att_w = nn.Linear(natt,1,bias=False) 

        self.att_u = nn.Linear(inp_size,natt,bias=False)
        self.att_h = nn.Linear(natt,natt,bias = False)
        
        self.no_att = no_att
    
    def forward(self, packed_batch,user_embs,item_embs, rev):
        
        self.rev = rev
        rnn_sents,_ = self.gru(packed_batch)
        enc_sents,len_s = torch.nn.utils.rnn.pad_packed_sequence(rnn_sents)

        hid_size = enc_sents.size(2) // 2
        sents_bilstm_enc = torch.cat([enc_sents[0, :, :hid_size], enc_sents[-1, :, hid_size:]],dim=1) # sentence bilstm output
        if self.no_att:
            return enc_sents.sum(0, True).squeeze(0), sents_bilstm_enc


        sum_ue = self.att_u(user_embs)  # W*e
        transformed_h = self.att_h(enc_sents.view(enc_sents.size(0) * enc_sents.size(1), -1))  # W*h
        summed = F.tanh(sum_ue + transformed_h.view(enc_sents.size()))  

        att = self.att_w(summed.view(summed.size(0) * summed.size(1), -1)).view(summed.size(0), summed.size(1)).transpose(0, 1)  #
        all_att = self._masked_softmax(att, self._list_to_bytemask(list(len_s))).transpose(0, 1)  # attW,sent #ai
        attended = all_att.unsqueeze(-1) * enc_sents  # ai*hi
        return attended.sum(0, True).squeeze(0), sents_bilstm_enc

    def _list_to_bytemask(self,l):
        mask = self._buffers['mask'].resize_(len(l),l[0]).fill_(1)

        for i,j in enumerate(l):
            if j != l[0]:
                mask[i,j:l[0]] = 0

        return mask
    
    def _masked_softmax(self,mat,mask):

        exp = torch.exp(mat) * Variable(mask,requires_grad=False)
        sum_exp = exp.sum(1,True)+0.0001

        softmax = exp/sum_exp.expand_as(exp)
        torch.set_printoptions(threshold=19000)
        d = {}
        batchsize = len(self.rev)
        global weight_dict
        for i in range(batchsize):
           
            r = self.rev[i].split()
            att = softmax[i]
            att_ = att.data.cpu().numpy()

            combination = tuple(zip(r,att_))
            if len(att) != 1:
                weight_dict[self.rev[i]] = combination

        if batchsize != 1:
            weight_dict = {}

        if batchsize == 1: 
            try:
                if list(weight_dict.values())[0].__len__() < 2:
                    weight_dict = {}
            except:
                weight_dict = {}

        return softmax

class EmojiSentimentClassification(nn.Module):

    def __init__(self, ntoken, emojis, nitems, num_class, emb_size, hid_size=50,no_att=False):
        super(EmojiSentimentClassification, self).__init__()

        self.emojis = emojis
        self.nitems = nitems
        self.embed = nn.Embedding(ntoken, emb_size, padding_idx=0)
        self.no_att = no_att

        self.emojis = nn.Embedding(emojis, emb_size)
        self.items = nn.Embedding(nitems, emb_size)

        self.word = AttentionalBiGRU(emb_size, emb_size//2,no_att=self.no_att)

        self.emb_size = emb_size
       
        self.lin_out = nn.Linear(emb_size*3,num_class)


    def set_emb_tensor(self,emb_tensor, emoji_tensor, zero_weight=False, auto_weight=True):
        self.emb_size = emb_tensor.size(-1)
        self.embed.weight.data = emb_tensor
        if auto_weight:
            return
        if zero_weight:
            self.emojis.weight.data = torch.zeros(self.nusers, self.emb_size)
            self.items.weight.data = torch.zeros(self.nitems, self.emb_size)
        else:
            self.emojis.weight.data = emoji_tensor
            self.items.weight.data = emoji_tensor
    def _reorder_sent(self,sents,stats):
        
        sort_r = sorted([(l,r,s,i) for i,(l,r,s) in enumerate(stats)], key=itemgetter(0,1,2)) #(len(r),r#,s#)

        r = [(l,r,s,i) for i,(l,r,s) in enumerate(stats)]

        builder = OrderedDict()
        
        for (l,r,s,i) in sort_r:
            if r not in builder:
                builder[r] = [i]
            else:
                builder[r].append(i)
                
        list_r = list(reversed(builder))
        
        revs = Variable(self._buffers["reviews"].resize_(len(builder),len(builder[list_r[0]]),sents.size(1)).fill_(0), requires_grad=False)
        lens = []
        review_order = []
        
        for i,x in enumerate(list_r):
            revs[i,0:len(builder[x]),:] = sents[builder[x],:]
            lens.append(len(builder[x]))
            review_order.append(x)

        real_order = sorted(range(len(review_order)), key=lambda k: review_order[k])
        
        return revs,lens,real_order,review_order
        
    
    def forward(self, batch_reviews,emojis,items,stats,rev):
        global weight_dict
        weight_dict = {}
        ls,lr,rn,sn = zip(*stats)
        emb_w = F.dropout(self.embed(batch_reviews),training=self.training)
        emb_emoji = F.dropout(self.emojis(emojis),training=self.training)
        emb_i = F.dropout(self.items(items),training=self.training)

        packed_sents = torch.nn.utils.rnn.pack_padded_sequence(emb_w, ls,batch_first=True)

        reordered_u = emb_emoji[rn,:]
        reordered_i = emb_i[rn,:]

        sent_embs, sents_bilstm_enc = self.word(packed_sents,reordered_u,reordered_i,rev)  #s, [h0,ht]

        if self.no_att:
            final_emb = sent_embs
            out = self.lin_out(final_emb)
        else:
            final_emb = torch.cat([sent_embs, sents_bilstm_enc, emb_emoji], dim = 1)
            out = self.lin_out(final_emb)
        return out



