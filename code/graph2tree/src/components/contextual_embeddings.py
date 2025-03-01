import torch.nn as nn
import torch
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer
from transformers import AutoTokenizer, AutoModel
import pdb

class BertEncoder(nn.Module):
	def __init__(self, bert_model = 'beomi/KcELECTRA-base',device = 'cuda:0 ', freeze_bert = False):
		super(BertEncoder, self).__init__()
		#bert_model = 'bert-base-uncased'
		bert_model = 'beomi/KcELECTRA-base'
		self.bert_layer = AutoModel.from_pretrained(bert_model)
		#self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_model, additional_special_tokens=['NUM'])
		self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_model)
		self.device = device
		
		if freeze_bert:
			for p in self.bert_layer.parameters():
				p.requires_grad = False
		
	def bertify_input(self, sentences):
		'''
		Preprocess the input sentences using bert tokenizer and converts them to a torch tensor containing token ids

		'''
		#Tokenize the input sentences for feeding into BERT
		# pdb.set_trace()
		all_tokens  = [['[CLS]'] + self.bert_tokenizer.tokenize(sentence) + ['[SEP]'] for sentence in sentences]

		index_retrieve = []
		for sent in all_tokens:
			cur_ls = []
			for j in range(1, len(sent)):
				if sent[j][0] == '#':
					continue
				else:
					cur_ls.append(j)
			index_retrieve.append(cur_ls)
		
		#Pad all the sentences to a maximum length
		input_lengths = [len(tokens) for tokens in all_tokens]
		max_length    = max(input_lengths)
		padded_tokens = [tokens + ['[PAD]' for _ in range(max_length - len(tokens))] for tokens in all_tokens]

		#Convert tokens to token ids
		token_ids = torch.tensor([self.bert_tokenizer.convert_tokens_to_ids(tokens) for tokens in padded_tokens]).to(self.device)

		#Obtain attention masks
		pad_token = self.bert_tokenizer.convert_tokens_to_ids('[PAD]')
		attn_masks = (token_ids != pad_token).long()

		return token_ids, attn_masks, input_lengths, index_retrieve

	def forward(self, sentences):
		'''
		Feed the batch of sentences to a BERT encoder to obtain contextualized representations of each token
		'''
		#Preprocess sentences
		token_ids, attn_masks, input_lengths, index_retrieve = self.bertify_input(sentences)

		#Feed through bert
		#cont_reps, _ = self.bert_layer(token_ids, attention_mask = attn_masks)
		res = self.bert_layer(token_ids, attention_mask = attn_masks)
		cont_reps = res.last_hidden_state

		return cont_reps, input_lengths, token_ids, index_retrieve

class RobertaEncoder(nn.Module):
	def __init__(self, roberta_model = 'roberta-base', device = 'cuda:0 ', freeze_roberta = False):
		super(RobertaEncoder, self).__init__()
		self.roberta_layer = RobertaModel.from_pretrained(roberta_model)
		self.roberta_tokenizer = RobertaTokenizer.from_pretrained(roberta_model)
		self.device = device
		
		if freeze_roberta:
			for p in self.roberta_layer.parameters():
				p.requires_grad = False
		
	def robertify_input(self, sentences):
		'''
		Preprocess the input sentences using roberta tokenizer and converts them to a torch tensor containing token ids

		'''
		# Tokenize the input sentences for feeding into RoBERTa
		all_tokens  = [['<s>'] + self.roberta_tokenizer.tokenize(sentence) + ['</s>'] for sentence in sentences]
		
		index_retrieve = []
		for sent in all_tokens:
			cur_ls = [1]
			for j in range(2, len(sent)):
				if sent[j][0] == '\u0120':
					cur_ls.append(j)
			index_retrieve.append(cur_ls)				
		
		# Pad all the sentences to a maximum length
		input_lengths = [len(tokens) for tokens in all_tokens]
		max_length    = max(input_lengths)
		padded_tokens = [tokens + ['<pad>' for _ in range(max_length - len(tokens))] for tokens in all_tokens]

		# Convert tokens to token ids
		token_ids = torch.tensor([self.roberta_tokenizer.convert_tokens_to_ids(tokens) for tokens in padded_tokens]).to(self.device)

		# Obtain attention masks
		pad_token = self.roberta_tokenizer.convert_tokens_to_ids('<pad>')
		attn_masks = (token_ids != pad_token).long()

		return token_ids, attn_masks, input_lengths, index_retrieve

	def forward(self, sentences):
		'''
		Feed the batch of sentences to a RoBERTa encoder to obtain contextualized representations of each token
		'''
		# Preprocess sentences
		token_ids, attn_masks, input_lengths, index_retrieve = self.robertify_input(sentences)

		# Feed through RoBERTa
		cont_reps, _ = self.roberta_layer(token_ids, attention_mask = attn_masks)

		return cont_reps, input_lengths, token_ids, index_retrieve