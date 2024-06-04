import argparse
import json
import os
import re
import time, datetime
import scipy
from typing import List, Literal
from collections import Counter
import tqdm
import numpy as np
import pandas as pd
import string
from sklearn import metrics
import math
# from Model.OpenNMTkpgrelease.onmt.inputters.keyphrase_dataset import infer_dataset_type, KP_DATASET_FIELDS, parse_src_fn
# from Model.OpenNMTkpgrelease.onmt.keyphrase.eval import compute_match_scores, run_classic_metrics, run_advanced_metrics
# from Model.OpenNMTkpgrelease.onmt.keyphrase.utils import if_present_duplicate_phrases, validate_phrases, print_predeval_result, gather_scores
# from Model.OpenNMTkpgrelease.onmt.utils.logging import init_logger
import LatvianStemmer
from nltk.stem.porter import *
from nltk.corpus import stopwords
from stempel import StempelStemmer
from bulstem.stem import BulStemmer
from nltk.stem import SnowballStemmer
from greek_stemmer import GreekStemmer
from eval.stemmers.stemm_sk import sk_stem
from eval.stemmers.czech_stemmer import cz_stem
from eval.stemmers.slovenian_stemmer import MirkoStemmer
from eval.stemmers.maltese_stemmer import MalteseStemmer
from eval.stemmers.croatian_stemmer import CroatianStemmer
from eval.stemmers.snowball_stemmer import extra_sbl_stemmer
from eval.stemmers.estonian_stemmer import estonian_shell_stemmer
from datasets import load_dataset

KPS_SEPARATOR = " ; "

# matplotlib.use('agg')
# import matplotlib.pyplot as plt

def stem_word_list(word_list, stemmer):
	if stemmer is not None:
		return [stemmer.stem(w.strip()) for w in word_list]
	else:
		return word_list
# import spacy
# spacy_nlp = spacy.load('en_core_web_sm')

def evaluate(src_list:List[str], tgt_list:List[List[str]], pred_list:List[List[str]],
			 unk_token:str, language:str,
			 logger=None, verbose=False,
			 report_path=None, tokenizer='spacy'):
	"""

	:param src_list: list of input str (abstracts)
	:param tgt_list: list of sequences of keywords (each element is a list of str)
	:param pred_list: list of lists of hypotheses (for the same instance, a list of one or more predicted keyword sequences are returned)
	:param unk_token: defined as "<unk>" in the models scripts (BARTe.py BARTf.py KeyBART.py scibart.py), but later defined as "unk" inside this function
			is it a typo? strongly recommend using ```tokenizer.unk_token```
			FYI, the following models
				facebook/bart-base
				facebook/mbart-large-5
				moussaKam/barthez
				moussaKam/mbarthez
			use the same tokens
				{'bos_token': '<s>',
				 'eos_token': '</s>',
				 'unk_token': '<unk>',
				 'sep_token': '</s>',
				 'pad_token': '<pad>',
				 'cls_token': '<s>',
				 'mask_token': '<mask>'}
	:param language: either en or fr
	:param logger:
	:param verbose:
	:param report_path: not sure if used, is it a directory ?
	:param tokenizer: 'spacy' 'split' or 'split_nopunc'
	:return: dictionary with keys (str) and scores (float)
						present_exact_f_score_hard@5
						present_exact_f_score@M
						absent_exact_f_score_hard@5
						absent_exact_f_score@M
	"""

	if language=='en':
		stemmer = PorterStemmer()
		import spacy
		spacy_nlp = spacy.load('en_core_web_sm')
	elif language=='fr':
		stemmer=SnowballStemmer('french')
	elif language=='mu':
		stemmer=None
	else:
		raise ValueError(f'Language {language} not recognized')
	# 'k' means the number of phrases in ground-truth, add 1,3 for openkp
	topk_range = [5, 'M']
	topk_range = [5, 10, 15, 'k', 'M', 1, 3]
	# topk_range = [5, 10, 15, 'k', 'M', 1, 3]
	absent_topk_range = [5, 'M']
	absent_topk_range = [5, 10, 15, 50, 'k', 'M']
	# 'precision_hard' and 'f_score_hard' mean that precision is calculated with denominator strictly as K (say 5 or 10), won't be lessened even number of preds is smaller
	metric_names = ['f_score', 'f_score_hard']
	metric_names = ['correct', 'precision', 'recall', 'f_score', 'precision_hard', 'f_score_hard'] # shown in function run_classic_metrics

	individual_score_dicts = []  # {'precision@5':[],'recall@5':[],'f1score@5':[], 'precision@10':[],'recall@10':[],'f1score@10':[]}
	gathered_score_dict = {}  # {'precision@5':[],'recall@5':[],'f1score@5':[], 'precision@10':[],'recall@10':[],'f1score@10':[]}
	# for i, (src_dict, tgt_dict, pred_dict) in tqdm.tqdm(enumerate(zip(src_list, tgt_list, pred_list))):
	for i, (src_sent, tgts, pred_seqs) in tqdm.tqdm(enumerate(zip(src_list, tgt_list, pred_list)), total=len(src_list)):
		"""
		1. Process each data example and predictions
		"""
		# pred_seqs = pred_dict["pred_sents"]
		if isinstance(pred_seqs, float) :
			pred_seqs = [] # sometimes, pred_seqs can be a float nan
		if len(pred_seqs) > 0 and isinstance(pred_seqs[0], str):
			pred_seqs = [p.split() for p in pred_seqs]
		# pred_idxs = pred_dict["preds"] if "preds" in pred_dict else None
		# pred_scores = pred_dict["pred_scores"] if "pred_scores" in pred_dict else None
		# copied_flags = pred_dict["copied_flags"] if "copied_flags" in pred_dict else None
		# @memray 20200410 add split_nopunc tokenization, spacy runs very slow
		if tokenizer == 'spacy':
			src_seq = [t.text for t in spacy_nlp(src_sent, disable=["textcat"])]
			tgt_seqs = [[t.text for t in spacy_nlp(p, disable=["textcat"])] for p in tgts]
			if len(pred_seqs) > 0 and isinstance(pred_seqs[0], str):
				pred_seqs = [[t.text for t in spacy_nlp(p, disable=["textcat"])] for p in pred_seqs]
			else:
				pred_seqs = [[t.text for t in spacy_nlp(' '.join(p), disable=["textcat"])] for p in pred_seqs]
			unk_token = 'unk'
		elif tokenizer == 'split':
			# src_seq = src_dict["src"].split()
			src_seq = src_sent.split()
			tgt_seqs = [t.split() for t in p]
			pred_seqs = pred_seqs
		elif tokenizer == 'split_nopunc':
			src_seq = [t for t in re.split(r'\W', src_sent) if len(t) > 0]
			tgt_seqs = [[t for t in re.split(r'\W', p) if len(t) > 0] for p in tgts]
			pred_seqs = [[t for t in re.split(r'\W', ' '.join(p)) if len(t) > 0] for p in pred_seqs]
			unk_token = 'unk'
		else:
			raise Exception('Unset or unsupported tokenizer for evaluation: %s' % str(tokenizer))
		# fds
		# 1st filtering, ignore phrases having <unk> and puncs
		valid_pred_flags = validate_phrases(pred_seqs, unk_token)
		# 2nd filtering: filter out phrases that don't appear in text, and keep unique ones after stemming
		present_pred_flags, _, duplicate_flags = if_present_duplicate_phrases(src_seq, pred_seqs, stemming=False, lowercase=True)
		# treat duplicates as invalid
		valid_pred_flags = valid_pred_flags * ~duplicate_flags if len(valid_pred_flags) > 0 else []
		valid_and_present_flags = valid_pred_flags * present_pred_flags if len(valid_pred_flags) > 0 else []
		valid_and_absent_flags = valid_pred_flags * ~present_pred_flags if len(valid_pred_flags) > 0 else []

		# compute match scores (exact, partial and mixed), for exact it's a list otherwise matrix
		match_scores_exact = compute_match_scores(tgt_seqs=tgt_seqs, pred_seqs=pred_seqs, do_lower=True, do_stem=True, type='exact', stemmer=stemmer)
		# match_scores_partial = compute_match_scores(tgt_seqs=tgt_seqs, pred_seqs=pred_seqs, do_lower=True, do_stem=False, type='ngram')
		# simply add full-text to n-grams might not be good as its contribution is not clear
		# match_scores_mixed = compute_match_scores(tgt_seqs=tgt_seqs, pred_seqs=pred_seqs, type='mixed')

		# split tgts by present/absent
		present_tgt_flags, _, _ = if_present_duplicate_phrases(src_seq, tgt_seqs, stemming=True, lowercase=True, stemmer=stemmer)
		present_tgts = [tgt for tgt, present in zip(tgt_seqs, present_tgt_flags) if present]
		absent_tgts = [tgt for tgt, present in zip(tgt_seqs, present_tgt_flags) if ~present]

		# filter out results of invalid preds
		valid_preds = [seq for seq, valid in zip(pred_seqs, valid_pred_flags) if valid]
		valid_present_pred_flags = present_pred_flags[valid_pred_flags]

		valid_match_scores_exact = match_scores_exact[valid_pred_flags]
		# valid_match_scores_partial = match_scores_partial[valid_pred_flags]
		# match_scores_mixed = match_scores_mixed[valid_pred_flags]

		# split preds by present/absent and exact/partial/mixed
		valid_present_preds = [pred for pred, present in zip(valid_preds, valid_present_pred_flags) if present]
		valid_absent_preds = [pred for pred, present in zip(valid_preds, valid_present_pred_flags) if ~present]
		if len(valid_present_pred_flags) > 0:
			present_exact_match_scores = valid_match_scores_exact[valid_present_pred_flags]
			# present_partial_match_scores = valid_match_scores_partial[valid_present_pred_flags][:, present_tgt_flags]
			# present_mixed_match_scores = match_scores_mixed[present_pred_flags][:, present_tgt_flags]
			absent_exact_match_scores = valid_match_scores_exact[~valid_present_pred_flags]
			# absent_partial_match_scores = valid_match_scores_partial[~valid_present_pred_flags][:, ~present_tgt_flags]
			# absent_mixed_match_scores = match_scores_mixed[~present_pred_flags][:, ~present_tgt_flags]
		else:
			present_exact_match_scores = []
			present_partial_match_scores = []
			# present_mixed_match_scores = []
			absent_exact_match_scores = []
			absent_partial_match_scores = []
			# absent_mixed_match_scores = []

		# assert len(valid_pred_seqs) == len(match_scores_exact) == len(present_pred_flags)
		# assert len(present_preds) == len(present_exact_match_scores) == len(present_partial_match_scores) == len(present_mixed_match_scores)
		# assert present_partial_match_scores.shape == present_mixed_match_scores.shape
		# assert len(absent_preds) == len(absent_exact_match_scores) == len(absent_partial_match_scores) == len(absent_mixed_match_scores)
		# assert absent_partial_match_scores.shape == absent_mixed_match_scores.shape

		"""
		2. Compute metrics
		"""
		# get the scores on different scores (for absent results, only recall matters)
		all_exact_results = run_classic_metrics(valid_match_scores_exact, valid_preds, tgt_seqs, metric_names, topk_range)
		present_exact_results = run_classic_metrics(present_exact_match_scores, valid_present_preds, present_tgts, metric_names, topk_range)
		absent_exact_results = run_classic_metrics(absent_exact_match_scores, valid_absent_preds, absent_tgts, metric_names, absent_topk_range)


		range_map=[]

		for i, hypo in enumerate(valid_preds):
			if i==50:
				break
			if hypo in tgt_seqs:
				range_map.append(i)
		if range_map==[]:
			MAP_50=0
		else:
			MAP_50=run_classic_metrics(valid_match_scores_exact, valid_preds, tgt_seqs, metric_names, range_map)
			MAP_50_metric=[]
			for key, value in MAP_50.items():
				if 'hard' in key:
					MAP_50_metric.append(value)
				else:
					continue
			MAP_50=np.mean(MAP_50_metric)

		# all_partial_results = run_classic_metrics(valid_match_scores_partial, valid_preds, tgt_seqs, metric_names, topk_range, type='partial')
		# present_partial_results = run_classic_metrics(present_partial_match_scores, valid_present_preds, present_tgts, metric_names, topk_range, type='partial')
		# absent_partial_results = run_classic_metrics(absent_partial_match_scores, valid_absent_preds, absent_tgts, metric_names, absent_topk_range, type='partial')
		# present_mixed_results = run_metrics(present_mixed_match_scores, present_preds, present_tgts, metric_names, topk_range, type='partial')
		# absent_mixed_results = run_metrics(absent_mixed_match_scores, absent_preds, absent_tgts, metric_names, absent_topk_range, type='partial')

		# all_exact_advanced_results = run_advanced_metrics(valid_match_scores_exact, valid_preds, tgt_seqs)
		# present_exact_advanced_results = run_advanced_metrics(present_exact_match_scores, valid_present_preds, present_tgts)
		# absent_exact_advanced_results = run_advanced_metrics(absent_exact_match_scores, valid_absent_preds, absent_tgts)
		# print(advanced_present_exact_results)
		# print(advanced_absent_exact_results)
		"""
		3. Gather scores
		"""
		eval_results_names = [
			'all_exact', #'all_partial',
			'present_exact', 'absent_exact',
			# 'present_partial', 'absent_partial',
			# 'present_mixed', 'absent_mixed'
			# 'all_exact_advanced', 'present_exact_advanced', 'absent_exact_advanced',
			]
		eval_results_list = [all_exact_results, #all_partial_results,
							 present_exact_results, absent_exact_results,
							 # present_partial_results, absent_partial_results,
							 # present_mixed_results, absent_mixed_results
							 # all_exact_advanced_results, present_exact_advanced_results, absent_exact_advanced_results
							]
		# update score_dict, appending new scores (results_list) to it
		individual_score_dict = {result_name: results for result_name, results in zip(eval_results_names, eval_results_list)}
		individual_score_dict['MAP@50']=MAP_50
		eval_results_list[0]['MAP@50']=MAP_50
		gathered_score_dict = gather_scores(gathered_score_dict, eval_results_names, eval_results_list)
		# add tgt/pred count for computing average performance on non-empty items
		stats_results_names = ['present_tgt_num', 'absent_tgt_num', 'present_pred_num', 'absent_pred_num', 'unique_pred_num', 'dup_pred_num', 'beam_num', 'beamstep_num']
		stats_results_list = [
						{'present_tgt_num': len(present_tgts)},
						{'absent_tgt_num': len(absent_tgts)},
						{'present_pred_num': len(valid_present_preds)},
						{'absent_pred_num': len(valid_absent_preds)},
						# TODO some stat should be calculated here since exhaustive/self-terminating makes difference
						{'unique_pred_num':  0},
						{'dup_pred_num': 0},
						{'beam_num': 10},
						{'beamstep_num': 0},
						# {'MAP@50': MAP_50},
						]
		for result_name, result_dict in zip(stats_results_names, stats_results_list):
			individual_score_dict[result_name] = result_dict[result_name]
		gathered_score_dict = gather_scores(gathered_score_dict, stats_results_names, stats_results_list)
		# gathered_score_dict['MAP@50'] = [MAP_50]
		# if len(gathered_score_dict['present_pred_num'])>1:
		# 	print(gathered_score_dict)
		# 	print(valid_preds)
		# 	print(tgt_seqs)
		# 	fds
		# individual_score_dicts.append(individual_score_dict)

		"""
		4. Print results if necessary
		"""
		# if verbose or report_file:
		# 	print_out = print_predeval_result(i, ' '.join(src_seq),
		# 									  tgt_seqs, present_tgt_flags,
		# 									  pred_seqs, #p#red_scores, pred_idxs, copied_flags,
		# 									  present_pred_flags, valid_pred_flags,
		# 									  valid_and_present_flags, valid_and_absent_flags,
		# 									  match_scores_exact, match_scores_partial,
		# 									  eval_results_names, eval_results_list, gathered_score_dict)
		#
		# 	if verbose:
		# 		if logger:
		# 			logger.info(print_out)
		# 		else:
		# 			print(print_out)
		#
		# 	if report_file:
		# 		report_file.write(print_out)

	# for k, v in score_dict.items():
	#     print('%s, num=%d, mean=%f' % (k, len(v), np.average(v)))

	# if report_file:
	# 	report_file.close(
	return gathered_score_dict

def validate_phrases(pred_seqs, unk_token):
	'''
	:param pred_seqs:
	:param src_str:
	:param oov:
	:param id2word:
	:param opt:
	:return:
	'''
	valid_flags = []

	for seq in pred_seqs:
		keep_flag = True

		if len(seq) == 0:
			keep_flag = False

		if keep_flag and any([w == unk_token for w in seq]):
			keep_flag = False

		if keep_flag and any([w == '.' or w == ',' for w in seq]):
			keep_flag = False

		valid_flags.append(keep_flag)

	return np.asarray(valid_flags)


def if_present_duplicate_phrases(src_seq, tgt_seqs, stemming=True, lowercase=True, stemmer=None):
	"""
	Check if each given target sequence verbatim appears in the source sequence
	:param src_seq:
	:param tgt_seqs:
	:param stemming:
	:param lowercase:
	:param check_duplicate:
	:return:
	"""
	if lowercase:
		src_seq = [w.lower() for w in src_seq]
	if stemming:
		src_seq = stem_word_list(src_seq, stemmer=stemmer)

	present_indices = []
	present_flags = []
	duplicate_flags = []
	phrase_set = set()  # some phrases are duplicate after stemming, like "model" and "models" would be same after stemming, thus we ignore the following ones

	for tgt_seq in tgt_seqs:
		if lowercase:
			tgt_seq = [w.lower() for w in tgt_seq]
		if stemming:
			tgt_seq = stem_word_list(tgt_seq, stemmer=stemmer)

		# check if the phrase appears in source text
		# iterate each word in source
		match_flag, match_pos_idx = if_present_phrase(src_seq, tgt_seq)

		# if it reaches the end of source and no match, means it doesn't appear in the source
		present_flags.append(match_flag)
		present_indices.append(match_pos_idx)

		# check if it is duplicate
		if '_'.join(tgt_seq) in phrase_set:
			duplicate_flags.append(True)
		else:
			duplicate_flags.append(False)
		phrase_set.add('_'.join(tgt_seq))

	assert len(present_flags) == len(present_indices)

	return np.asarray(present_flags), \
		   np.asarray(present_indices), \
		   np.asarray(duplicate_flags)


def if_present_phrase(src_str_tokens, phrase_str_tokens):
	"""

	:param src_str_tokens: a list of strings (words) of source text
	:param phrase_str_tokens: a list of strings (words) of a phrase
	:return:
	"""
	match_flag = False
	match_pos_idx = -1
	for src_start_idx in range(len(src_str_tokens) - len(phrase_str_tokens) + 1):
		match_flag = True
		# iterate each word in target, if one word does not match, set match=False and break
		for seq_idx, seq_w in enumerate(phrase_str_tokens):
			src_w = src_str_tokens[src_start_idx + seq_idx]
			if src_w != seq_w:
				match_flag = False
				break
		if match_flag:
			match_pos_idx = src_start_idx
			break

	return match_flag, match_pos_idx


def gather_scores(gathered_scores, results_names, results_dicts):
	for result_name, result_dict in zip(results_names, results_dicts):
		for metric_name, score in result_dict.items():
			if metric_name.endswith('_num'):
				# if it's 'present_tgt_num' or 'absent_tgt_num', leave as is
				field_name = result_name
			else:
				# if it's other score like 'precision@5' is renamed to like 'present_exact_precision@'
				field_name = result_name + '_' + metric_name

			if field_name not in gathered_scores:
				gathered_scores[field_name] = []

			gathered_scores[field_name].append(score)
	return gathered_scores

def kp_results_to_str(results_dict):
	"""
	return ">> ROUGE(1/2/3/L/SU4): {:.2f}/{:.2f}/{:.2f}/{:.2f}/{:.2f}".format(
		results_dict["rouge_1_f_score"] * 100,
		results_dict["rouge_2_f_score"] * 100,
		results_dict["rouge_3_f_score"] * 100,
		results_dict["rouge_l_f_score"] * 100,
		results_dict["rouge_su*_f_score"] * 100)
	"""
	summary_dict = {}
	for k,v in results_dict.items():
		summary_dict[k] = np.average(v)

	return json.dumps(summary_dict)


def baseline_pred_loader(pred_path, model_name):
	pred_dict_list = []

	if model_name in ['tfidf', 'textrank', 'singlerank', 'expandrank', 'maui', 'kea']:
		doc_list = [file_name for file_name in os.listdir(pred_path) if file_name.endswith('txt.phrases')]
		doc_list = sorted(doc_list, key=lambda k: int(k[:k.index('.txt.phrases')]))
		for doc_name in doc_list:
			doc_path = os.path.join(pred_path, doc_name)
			pred_dict = {}
			pred_dict['pred_sents'] = []

			for l in open(doc_path, 'r').readlines():
				pred_dict['pred_sents'].append(l.lower().split())
			pred_dict_list.append(pred_dict)
	else:
		raise NotImplementedError

	return pred_dict_list


def keyphrase_eval(datasplit_name, src_path, tgt_path, pred_path,
				   unk_token='<unk>', verbose=False, logger=None,
				   report_path=None, model_name='nn',
				   tokenizer=None):
	# change data loader to iterator, otherwise it consumes more than 64gb RAM
	# check line numbers first
	dataset_name = '_'.join(datasplit_name.split('_')[: -1])
	split_name = datasplit_name.split('_')[-1]
	dataset_name = dataset_name.strip().lower()
	src_line_number = sum([1 for _ in open(src_path, "r")])
	tgt_line_number = sum([1 for _ in open(tgt_path, "r")])
	if model_name == 'nn':
		pred_line_number = sum([1 for _ in open(pred_path, "r")])
	else:
		pred_line_number = len(baseline_pred_loader(pred_path, model_name))

	logger.info("pred file=%s" % (pred_path))
	logger.info("#(src)=%d, #(tgt)=%d, #(pred)=%d" % (src_line_number, tgt_line_number, pred_line_number))
	if src_line_number == tgt_line_number == pred_line_number:
		src_data = [json.loads(l) for l in open(src_path, "r")]
		tgt_data = [json.loads(l) for l in open(tgt_path, "r")]

		# Load from the json-format raw data, preprocess the src and tgt
		if src_path.endswith('json') or src_path.endswith('jsonl'):
			assert src_path == tgt_path, \
				'src and tgt should be from the same raw file: \n\tsrc_path: %s \n\ttgt_path: %s' % (src_path, tgt_path)
			dataset_type = infer_dataset_type(src_path)
			title_field, text_field, keyword_field, _ = KP_DATASET_FIELDS[dataset_type]

			for src_ex, tgt_ex in zip(src_data, tgt_data):
				src_str = parse_src_fn(src_ex, title_field, text_field)
				if isinstance(tgt_ex[keyword_field], str):
					tgt_kps = tgt_ex[keyword_field].split(';')
				else:
					tgt_kps = tgt_ex[keyword_field]

				src_ex['src'] = src_str
				tgt_ex['tgt'] = tgt_kps
		else:
			raise Exception('Currently only support json/jsonl data format: %s' % src_path)

		if model_name == 'nn':
			pred_data = [json.loads(l) for l in open(pred_path, "r")]
		else:
			pred_data = baseline_pred_loader(pred_path, model_name)
		start_time = time.time()
		results_dict = evaluate(src_data, tgt_data, pred_data,
								unk_token=unk_token,
								logger=logger, verbose=verbose,
								report_path=report_path,
								tokenizer=tokenizer)
		total_time = time.time() - start_time
		logger.info("Total evaluation time (s): %f" % total_time)

		return results_dict
	else:
		logger.error("")
		return None


def summarize_scores(score_dict, ckpt_name,
					 exp_name=None, pred_name=None, dataset_name=None,
					 eval_file_path=None, pred_file_path=None, step=None):
	avg_dict = {}
	avg_dict['checkpoint_name'] = ckpt_name
	avg_dict['exp_name'] = exp_name
	avg_dict['pred_name'] = pred_name
	avg_dict['test_dataset'] = dataset_name
	avg_dict['eval_file_path'] = eval_file_path
	avg_dict['pred_file_path'] = pred_file_path
	if step is not None:
		avg_dict['step'] = step
	elif ckpt_name.find('_') > 0:
		avg_dict['step'] = ckpt_name.rsplit('_')[-1]
	else:
		avg_dict['step'] = ''

	# doc stat
	avg_dict['#doc'] = len(score_dict['present_tgt_num'])
	avg_dict['#pre_doc'] = len([x for x in score_dict['present_tgt_num'] if x > 0])
	avg_dict['#ab_doc'] = len([x for x in score_dict['absent_tgt_num'] if x > 0])

	# tgt stat
	if 'present_tgt_num' in score_dict and 'absent_tgt_num' in score_dict:
		avg_dict['#tgt'] = np.average(score_dict['present_tgt_num']) + np.average(score_dict['absent_tgt_num'])
		avg_dict['#pre_tgt'] = np.average(score_dict['present_tgt_num'])
		avg_dict['#ab_tgt'] = np.average(score_dict['absent_tgt_num'])
	else:
		avg_dict['#tgt'] = 0
		avg_dict['#pre_tgt'] = 0
		avg_dict['#ab_tgt'] = 0

	# pred stat
	if 'present_pred_num' in score_dict and 'absent_pred_num' in score_dict:
		avg_dict['#pred'] = np.average(score_dict['present_pred_num']) + np.average(score_dict['absent_pred_num'])
		avg_dict['#pre_pred'] = np.average(score_dict['present_pred_num'])
		avg_dict['#ab_pred'] = np.average(score_dict['absent_pred_num'])
	else:
		avg_dict['#pred'] = 0
		avg_dict['#pre_pred'] = 0
		avg_dict['#ab_pred'] = 0

	avg_dict['#uni_pred'] = np.average(score_dict['unique_pred_num']) if 'unique_pred_num' in score_dict else 0
	avg_dict['#dup_pred'] = np.average(score_dict['dup_pred_num']) if 'dup_pred_num' in score_dict else 0
	avg_dict['#beam'] = np.average(score_dict['beam_num']) if 'beam_num' in score_dict else 0
	avg_dict['#beamstep'] = np.average(score_dict['beamstep_num']) if 'beamstep_num' in score_dict else 0

	# remove meta stats from score_dict
	if 'unique_pred_num' in score_dict: del score_dict['present_tgt_num']
	if 'absent_tgt_num' in score_dict: del score_dict['absent_tgt_num']
	if 'present_pred_num' in score_dict: del score_dict['present_pred_num']
	if 'absent_pred_num' in score_dict: del score_dict['absent_pred_num']
	if 'unique_pred_num' in score_dict: del score_dict['unique_pred_num']
	if 'dup_pred_num' in score_dict: del score_dict['dup_pred_num']
	if 'beam_num' in score_dict: del score_dict['beam_num']
	if 'beamstep_num' in score_dict: del score_dict['beamstep_num']

	# average scores of each metric
	for score_name, score_list in score_dict.items():
		# number of correct phrases
		if score_name.find('correct') > 0:
			# only keep exact results (partial count is trivial)
			if score_name.find('exact') > 0:
				avg_dict[score_name] = np.sum(score_list)
			continue

		# various scores (precision, recall, f-score)
		# NOTE! here can be tricky, we can average over all data examples or just valid examples
		#  in empirical paper, we use the former, to keep it consistent and simple
		'''
		if score_name.startswith('all') or score_name.startswith('present'):
			tmp_scores = [score for score, num in zip(score_list, present_tgt_num) if num > 0]
			avg_dict[score_name] = np.average(tmp_scores)
		elif score_name.startswith('absent'):
			tmp_scores = [score for score, num in zip(score_list, absent_tgt_num) if num > 0]
			avg_dict[score_name] = np.average(tmp_scores)
		else:
			logger.error("NotImplementedError: found key %s" % score_name)
			raise NotImplementedError
		'''
		avg_dict[score_name] = np.average(score_list)

	columns = list(avg_dict.keys())
	# print(columns)
	summary_df = pd.DataFrame.from_dict(avg_dict, orient='index').transpose()[columns]
	# print('\n')
	# print(list(summary_df.columns))
	# input()

	return summary_df


def gather_eval_results(eval_root_dir, report_csv_dir=None, tokenizer=None, empirical_result=False):
	dataset_scores_dict = {}
	assert tokenizer is not None
	evals_to_skip = set()
	if report_csv_dir:
		# load previous reports
		for report_csv_file in os.listdir(report_csv_dir):
			if not report_csv_file.endswith('.%s.csv' % tokenizer): continue
			dataset_name = report_csv_file.split('.')[0] # truncate 'tokenizer.csv'
			prev_df = pd.read_csv(os.path.join(report_csv_dir, report_csv_file))
			prev_df = prev_df.loc[:, ~prev_df.columns.str.contains('^Unnamed')]

			dataset_scores_dict[dataset_name] = prev_df
			for eval_path in prev_df.eval_file_path:
				evals_to_skip.add(eval_path)

	eval_suffix = '.%s.eval' % tokenizer
	total_file_num = len([file for subdir, dirs, files in os.walk(eval_root_dir)
						  for file in files if file.endswith(eval_suffix)])
	file_count = 0

	for subdir, dirs, files in os.walk(eval_root_dir):
		for file in files:
			if not file.endswith(eval_suffix): continue
			file_count += 1
			if file_count % 10 == 0: print("file_count/file_num=%d/%d" % (file_count, total_file_num))

			eval_file_path = os.path.join(subdir, file)
			pred_file_path = eval_file_path[: -len(eval_suffix)]+'.pred' # might be a very bad way
			if eval_file_path in evals_to_skip: continue
			if not os.path.exists(pred_file_path):
				# only count ones that both pred/eval exist, and remove some leftover files
				if os.path.exists(eval_file_path): os.remove(eval_file_path)
				report_file_path = eval_file_path[:-4]+'report'
				if os.path.exists(report_file_path): os.remove(report_file_path)
				continue

			if empirical_result:
				# legacy result
				exp_step_name = subdir.strip('/')[subdir.strip('/').rfind('/') + 1:]
				exp_name, step = exp_step_name.split('_step_')
				dataset_name = file[: file.find(eval_suffix)]
				ckpt_name = 'checkpoint_step_%s' % step
				pred_name = 'meng17-one2seq-beam50-maxlen40'  # very hard-coded
			else:
				file_name = file[: file.find(eval_suffix)]
				ckpt_name = file_name[: file.rfind('-')] if file.find('-') > 0 else file_name
				# exp_dirname = re.search('.*/(.*?)/outputs', subdir).group(1)
				# exp_name = exp_dirname.split('/')[1]
				exp_name = re.search('.*/(.*?)/outputs', subdir).group(1)
				pred_name = re.search('outputs/(.*?)/pred', subdir).group(1) # very hard-coded
				dataset_name = file_name[file.rfind('-') + 1: ]
				dataset_name = dataset_name[5:] if dataset_name.startswith('data_') else dataset_name
				step = None

			# key is dataset name, value is a dict whose key is metric name and value is a list of floats
			try:
				score_dict = json.load(open(eval_file_path, 'r'))
			except:
				print('error while loading %s' % eval_file_path)
				continue
			# ignore scores where no tgts available and return the average
			score_df = summarize_scores(score_dict,
										ckpt_name, exp_name, pred_name, dataset_name,
										eval_file_path, pred_file_path, step=step)

			# print(df_key)
			if dataset_name in dataset_scores_dict:
				dataset_scores_dict[dataset_name] = dataset_scores_dict[dataset_name].append(score_df)
			else:
				dataset_scores_dict[dataset_name] = score_df

		#     if file_count > 20:
		#         break
		#
		# if file_count > 20:
		#     break

	if report_csv_dir:
		for dataset_name, score_df in dataset_scores_dict.items():
			report_csv_path = os.path.join(report_csv_dir, dataset_name + '.%s.csv' % tokenizer)
			print("Writing summary to: %s" % (report_csv_path))
			score_df = score_df.sort_values(by=['exp_name', 'step'])
			score_df.to_csv(report_csv_path, index=False)
			# print(score_df.to_csv(index=False))

	return dataset_scores_dict

def init_opt():

	parser = argparse.ArgumentParser()
	# Input/output options
	parser.add_argument('--data', '-data', required=True,
						help="Path to the source/target file of groundtruth data.")
	parser.add_argument('--pred_dir', '-pred_dir', required=True,
						help="Directory to pred folders, each folder contains .pred files, each line is a JSON dict about predicted keyphrases.")
	parser.add_argument('--output_dir', '-output_dir',
						help="Path to output log/results.")
	parser.add_argument('--unk_token', '-unk_token', default="<unk>",
						help=".")
	parser.add_argument('--verbose', '-v', action='store_true',
						help=".")
	parser.add_argument('-testsets', nargs='+', type=str, default=["inspec", "krapivin", "nus", "semeval", "duc"], help='Specify datasets to test on')

	opt = parser.parse_args()

	return opt


def compute_match_scores(tgt_seqs, pred_seqs, do_lower=True, do_stem=False, type='exact', stemmer=None):
	'''
	If type='exact', returns a list of booleans indicating if a pred has a matching tgt
	If type='partial', returns a 2D matrix, each value v_ij is a float in range of [0,1]
		indicating the (jaccard) similarity between pred_i and tgt_j
	:param tgt_seqs:
	:param pred_seqs:
	:param do_stem:
	:param topn:
	:param type: 'exact' or 'partial'
	:return:
	'''
	# do processing to baseline predictions
	if type == "exact":
		match_score = np.zeros(shape=(len(pred_seqs)), dtype='float32')
	else:
		match_score = np.zeros(shape=(len(pred_seqs), len(tgt_seqs)), dtype='float32')

	target_number = len(tgt_seqs)
	predicted_number = len(pred_seqs)

	metric_dict = {'target_number': target_number, 'prediction_number': predicted_number, 'correct_number': match_score}

	# convert target index into string
	if do_lower:
		tgt_seqs = [[w.lower() for w in seq] for seq in tgt_seqs]
		pred_seqs = [[w.lower() for w in seq] for seq in pred_seqs]
	if do_stem and stemmer is not None:
		tgt_seqs = [stem_word_list(seq, stemmer=stemmer) for seq in tgt_seqs]
		pred_seqs = [stem_word_list(seq, stemmer=stemmer) for seq in pred_seqs]

	for pred_id, pred_seq in enumerate(pred_seqs):
		if type == 'exact':
			match_score[pred_id] = 0
			for true_id, true_seq in enumerate(tgt_seqs):
				match = True
				if len(pred_seq) != len(true_seq):
					continue
				for pred_w, true_w in zip(pred_seq, true_seq):
					# if one two words are not same, match fails
					if pred_w != true_w:
						match = False
						break
				# if every word in pred_seq matches one true_seq exactly, match succeeds
				if match:
					match_score[pred_id] = 1
					break
		elif type == 'ngram':
			# use jaccard coefficient as the similarity of partial match (1+2 grams)
			pred_seq_set = set(pred_seq)
			pred_seq_set.update(set([pred_seq[i]+'_'+pred_seq[i+1] for i in range(len(pred_seq)-1)]))
			for true_id, true_seq in enumerate(tgt_seqs):
				true_seq_set = set(true_seq)
				true_seq_set.update(set([true_seq[i]+'_'+true_seq[i+1] for i in range(len(true_seq)-1)]))
				if float(len(set.union(*[set(true_seq_set), set(pred_seq_set)]))) > 0:
					similarity = len(set.intersection(*[set(true_seq_set), set(pred_seq_set)])) \
							  / float(len(set.union(*[set(true_seq_set), set(pred_seq_set)])))
				else:
					similarity = 0.0
				match_score[pred_id, true_id] = similarity
		elif type == 'mixed':
			# similar to jaccard, but addtional to 1+2 grams we also put in the full string, serves like an exact+partial surrogate
			pred_seq_set = set(pred_seq)
			pred_seq_set.update(set([pred_seq[i]+'_'+pred_seq[i+1] for i in range(len(pred_seq)-1)]))
			pred_seq_set.update(set(['_'.join(pred_seq)]))
			for true_id, true_seq in enumerate(tgt_seqs):
				true_seq_set = set(true_seq)
				true_seq_set.update(set([true_seq[i]+'_'+true_seq[i+1] for i in range(len(true_seq)-1)]))
				true_seq_set.update(set(['_'.join(true_seq)]))
				if float(len(set.union(*[set(true_seq_set), set(pred_seq_set)]))) > 0:
					similarity = len(set.intersection(*[set(true_seq_set), set(pred_seq_set)])) \
							  / float(len(set.union(*[set(true_seq_set), set(pred_seq_set)])))
				else:
					similarity = 0.0
				match_score[pred_id, true_id] = similarity

		elif type == 'bleu':
			# account for the match of subsequences, like n-gram-based (BLEU) or LCS-based
			# n-grams precision doesn't work that well
			for true_id, true_seq in enumerate(tgt_seqs):
				match_score[pred_id, true_id] = bleu(pred_seq, [true_seq], [0.7, 0.3, 0.0])

	return match_score


def run_classic_metrics(match_list, pred_list, tgt_list, score_names, topk_range, type='exact'):
	"""
	Return a dict of scores containing len(score_names) * len(topk_range) items
	score_names and topk_range actually only define the names of each score in score_dict.
	:param match_list:
	:param pred_list:
	:param tgt_list:
	:param score_names:
	:param topk_range:
	:param type: exact or partial
	:return:
	"""
	score_dict = {}
	if len(tgt_list) == 0:
		for topk in topk_range:
			for score_name in score_names:
				score_dict['{}@{}'.format(score_name, topk)] = 0.0
		return score_dict

	assert len(match_list) == len(pred_list)
	for topk in topk_range:
		if topk == 'k':
			cutoff = len(tgt_list)
		elif topk == 'M':
			cutoff = len(pred_list)
		else:
			cutoff = topk

		# print(pred_list)
		if len(pred_list) > cutoff:
			pred_list_k = np.asarray(pred_list[:cutoff], dtype=object)
			match_list_k = match_list[:cutoff]
		else:
			pred_list_k = np.asarray(pred_list, dtype=object)
			match_list_k = match_list

		if type == 'partial':
			cost_matrix = np.asarray(match_list_k, dtype=float)
			if len(match_list_k) > 0:
				# convert to a negative matrix because linear_sum_assignment() looks for minimal assignment
				row_ind, col_ind = scipy.optimize.linear_sum_assignment(-cost_matrix)
				match_list_k = cost_matrix[row_ind, col_ind]
				overall_cost = cost_matrix[row_ind, col_ind].sum()
			'''
			print("\n%d" % topk)
			print(row_ind, col_ind)
			print("Pred" + str(np.asarray(pred_list)[row_ind].tolist()))
			print("Target" + str(tgt_list))
			print("Maximum Score: %f" % overall_cost)

			print("Pred list")
			for p_id, (pred, cost) in enumerate(zip(pred_list, cost_matrix)):
				print("\t%d \t %s - %s" % (p_id, pred, str(cost)))
			'''

		# Micro-Averaged Method
		correct_num = int(sum(match_list_k))
		# Precision, Recall and F-score, with flexible cutoff (if number of pred is smaller)
		micro_p = float(sum(match_list_k)) / float(len(pred_list_k)) if len(pred_list_k) > 0 else 0.0
		micro_r = float(sum(match_list_k)) / float(len(tgt_list)) if len(tgt_list) > 0 else 0.0

		if micro_p + micro_r > 0:
			micro_f1 = float(2 * (micro_p * micro_r)) / (micro_p + micro_r)
		else:
			micro_f1 = 0.0
		# F-score, with a hard cutoff on precision, offset the favor towards fewer preds
		micro_p_hard = float(sum(match_list_k)) / cutoff if len(pred_list_k) > 0 else 0.0
		if micro_p_hard + micro_r > 0:
			micro_f1_hard = float(2 * (micro_p_hard * micro_r)) / (micro_p_hard + micro_r)
		else:
			micro_f1_hard = 0.0

		for score_name, v in zip(['correct', 'precision', 'recall', 'f_score', 'precision_hard', 'f_score_hard'], [correct_num, micro_p, micro_r, micro_f1, micro_p_hard, micro_f1_hard]):
			score_dict['{}@{}'.format(score_name, topk)] = v

	# return only the specified scores
	return_scores = {}
	for topk in topk_range:
		for score_name in score_names:
			return_scores['{}@{}'.format(score_name, topk)] = score_dict['{}@{}'.format(score_name, topk)]

	return return_scores


# def run_advanced_metrics(match_scores, pred_list, tgt_list):
# 	score_dict = {}
# 	corrects, precisions, recalls, fscores = compute_PRF1(match_scores, pred_list, tgt_list)
# 	auc = compute_PR_AUC(precisions, recalls)
# 	ap = compute_AP(match_scores, precisions, tgt_list)
# 	mrr = compute_MRR(match_scores)
# 	sadr = compute_SizeAdjustedDiscountedRecall(match_scores, tgt_list)
# 	ndcg = compute_NormalizedDiscountedCumulativeGain(match_scores, tgt_list)
# 	# alpha_ndcg_5 = compute_alphaNormalizedDiscountedCumulativeGain(pred_list, tgt_list, k=5, alpha=0.5)
# 	# alpha_ndcg_10 = compute_alphaNormalizedDiscountedCumulativeGain(pred_list, tgt_list, k=10, alpha=0.5)
#
# 	score_dict['auc'] = auc
# 	score_dict['ap'] = ap
# 	score_dict['mrr'] = mrr
# 	score_dict['sadr'] = sadr
# 	score_dict['ndcg'] = ndcg
# 	# score_dict['alpha_ndcg@5'] = alpha_ndcg_5
# 	# score_dict['alpha_ndcg@10'] = alpha_ndcg_10
#
# 	# print('\nMatch[#=%d]=%s' % (len(match_scores), str(match_scores)))
# 	# print('Accum Corrects=' + str(corrects))
# 	# print('P@x=' + str(precisions))
# 	# print('R@x=' + str(recalls))
# 	# print('F-score@x=' + str(fscores))
# 	#
# 	# print('F-score@5=%f' % fscores[4])
# 	# print('F-score@10=%f' % (fscores[9] if len(fscores) > 9 else -9999))
# 	# print('F-score@O=%f' % fscores[len(tgt_list) - 1])
# 	# print('F-score@M=%f' % fscores[len(match_scores) - 1])
# 	#
# 	# print('AUC=%f' % auc)
# 	# print('AP=%f' % ap)
# 	# print('MRR=%f' % mrr)
# 	# print('SADR=%f' % sadr)
# 	# print('nDCG=%f' % ndcg)
# 	# print('α-nDCG@5=%f' % alpha_ndcg_5)
# 	# print('α-nDCG@10=%f' % alpha_ndcg_10)
#
# 	return score_dict
#
# def compute_PRF1(match_scores, preds, tgts):
# 	corrects, precisions, recalls, fscores = [], [], [], []
#
# 	for pred_id, score in enumerate(match_scores):
# 		_corr = corrects[-1] + score if len(corrects) > 0 else score
# 		_p = _corr / (pred_id + 1) if pred_id + 1 > 0 else 0.0
# 		_r = _corr / len(tgts) if len(tgts) > 0 else 0.0
# 		_f1 = float(2 * (_p * _r)) / (_p + _r) if (_p + _r) > 0 else 0.0
# 		corrects += [_corr]
# 		precisions += [_p]
# 		recalls += [_r]
# 		fscores += [_f1]
#
# 	return corrects, precisions, recalls, fscores

#
# def compute_MRR(match_scores):
# 	# A modified mean reciprocal rank for KP eval
# 	# MRR in IR uses the rank of first correct result. We use the rank of all correctly recalled results.
# 	# But it doesn't consider the missing predictions, so it's a precision-like metric
# 	mrr = 0.0
# 	count = 0.0
#
# 	for idx, match_score in enumerate(match_scores):
# 		if match_score == 0.0:
# 			continue
# 		mrr += match_score / (idx + 1)
# 		count += 1.0
#
# 	if count > 0:
# 		mrr /= count
# 	else:
# 		mrr = 0.0
# 	return mrr
#
#
# def compute_AP(match_scores, precisions, tgts):
# 	# Average Precision: Note that the average is over all relevant documents and the relevant documents not retrieved get a precision score of zero.
# 	# Updated on March 4, 2020. Previously we average over the number of correct predictions.
# 	ap = 0.0
# 	tgt_count = len(tgts)
#
# 	for idx, (match_score, precision) in enumerate(zip(match_scores, precisions)):
# 		if match_score == 0.0:
# 			continue
# 		ap += precision
#
# 	if tgt_count > 0:
# 		ap /= tgt_count
# 	else:
# 		ap = 0.0
# 	return ap


# def compute_PR_AUC(precisions, recalls):
# 	# we need to pad two values as the begin/end point of the curve
# 	p = [1.0] + precisions + [0.0]
# 	r = [0.0] + recalls + [(recalls[-1] if len(recalls) > 0 else 0.0)]
# 	pr_auc = metrics.auc(r, p)
#
# 	return pr_auc

#
# def compute_SizeAdjustedDiscountedRecall(match_scores, tgts):
# 	# add a log2(pos-num_tgt+2) discount to correct predictions out of the top-k list
# 	cumulated_gain = 0.0
# 	num_tgts = len(tgts)
#
# 	for idx, match_score in enumerate(match_scores):
# 		if match_score == 0.0:
# 			continue
# 		if idx + 1 > num_tgts:
# 			gain = 1.0 / math.log(idx - num_tgts + 3, 2)
# 		else:
# 			gain = 1.0
# 		# print('gain@%d=%f' % (idx + 1, gain))
# 		cumulated_gain += gain
#
# 	if num_tgts > 0:
# 		ndr = cumulated_gain / num_tgts
# 	else:
# 		ndr = 0.0
#
# 	return ndr
#
#
# def compute_NormalizedDiscountedCumulativeGain(match_scores, tgts):
# 	# add a positional discount to all predictions
# 	def _compute_dcg(match_scores):
# 		cumulated_gain = 0.0
# 		for idx, match_score in enumerate(match_scores):
# 			gain = match_score / math.log(idx + 2, 2)
# 			#             print('gain@%d=%f' % (idx + 1, gain))
# 			cumulated_gain += gain
# 		return cumulated_gain
#
# 	num_tgts = len(tgts)
# 	assert sum(match_scores) <= num_tgts, "Sum of relevance scores shouldn't exceed number of targets."
# 	if num_tgts > 0:
# 		dcg = _compute_dcg(match_scores)
# 		#         print('DCG=%f' % dcg)
# 		idcg = _compute_dcg([1.0] * num_tgts)
# 		#         print('IDCG=%f' % idcg)
# 		ndcg = dcg / idcg
# 	else:
# 		ndcg = 0.0
#
# 	#     print('nDCG=%f' % ndcg)
# 	return ndcg




# def f1_score(prediction, ground_truth):
# 	# both prediction and grount_truth should be list of words
# 	common = Counter(prediction) & Counter(ground_truth)
# 	num_same = sum(common.values())
# 	if num_same == 0:
# 		return 0
# 	precision = 1.0 * num_same / len(prediction) if len(prediction) > 0 else 0.0
# 	recall = 1.0 * num_same / len(ground_truth) if len(ground_truth) > 0 else 0.0
# 	f1 = (2 * precision * recall) / (precision + recall) if len(precision + recall) > 0 else 0.0
# 	return f1


# def macro_averaged_score(precisionlist, recalllist):
# 	precision = np.average(precisionlist)
# 	recall = np.average(recalllist)
# 	f_score = 0
# 	if(precision or recall):
# 		f_score = round((2 * (precision * recall)) / (precision + recall), 2)
# 	return precision, recall, f_score


# def self_redundancy(_input):
# 	# _input shoule be list of list of words
# 	if len(_input) == 0:
# 		return None
# 	_len = len(_input)
# 	scores = np.ones((_len, _len), dtype="float32") * -1.0
# 	for i in range(_len):
# 		for j in range(_len):
# 			if scores[i][j] != -1:
# 				continue
# 			elif i == j:
# 				scores[i][j] = 0.0
# 			else:
# 				f1 = f1_score(_input[i], _input[j])
# 				scores[i][j] = f1
# 				scores[j][i] = f1
# 	res = np.max(scores, 1)
# 	res = np.mean(res)
# 	return res

#
# def eval_and_print(src_text, tgt_kps, pred_kps, pred_scores, unk_token='<unk>', return_eval=False):
# 	src_seq = [t for t in re.split(r'\W', src_text) if len(t) > 0]
# 	tgt_seqs = [[t for t in re.split(r'\W', p) if len(t) > 0] for p in tgt_kps]
# 	pred_seqs = [[t for t in re.split(r'\W', p) if len(t) > 0] for p in pred_kps]
#
# 	topk_range = ['k', 10]
# 	absent_topk_range = [50, 'M']
# 	metric_names = ['f_score']
#
# 	# 1st filtering, ignore phrases having <unk> and puncs
# 	valid_pred_flags = validate_phrases(pred_seqs, unk_token)
# 	# 2nd filtering: filter out phrases that don't appear in text, and keep unique ones after stemming
# 	present_pred_flags, _, duplicate_flags = if_present_duplicate_phrases(src_seq, pred_seqs)
# 	# treat duplicates as invalid
# 	valid_pred_flags = valid_pred_flags * ~duplicate_flags if len(valid_pred_flags) > 0 else []
# 	valid_and_present_flags = valid_pred_flags * present_pred_flags if len(valid_pred_flags) > 0 else []
# 	valid_and_absent_flags = valid_pred_flags * ~present_pred_flags if len(valid_pred_flags) > 0 else []
#
# 	# compute match scores (exact, partial and mixed), for exact it's a list otherwise matrix
# 	match_scores_exact = compute_match_scores(tgt_seqs=tgt_seqs, pred_seqs=pred_seqs,
# 											  do_lower=True, do_stem=False, type='exact')
# 	# split tgts by present/absent
# 	present_tgt_flags, _, _ = if_present_duplicate_phrases(src_seq, tgt_seqs)
# 	present_tgts = [tgt for tgt, present in zip(tgt_seqs, present_tgt_flags) if present]
# 	absent_tgts = [tgt for tgt, present in zip(tgt_seqs, present_tgt_flags) if ~present]
#
# 	# filter out results of invalid preds
# 	valid_preds = [seq for seq, valid in zip(pred_seqs, valid_pred_flags) if valid]
# 	valid_present_pred_flags = present_pred_flags[valid_pred_flags]
#
# 	valid_match_scores_exact = match_scores_exact[valid_pred_flags]
#
# 	# split preds by present/absent and exact/partial/mixed
# 	valid_present_preds = [pred for pred, present in zip(valid_preds, valid_present_pred_flags) if present]
# 	valid_absent_preds = [pred for pred, present in zip(valid_preds, valid_present_pred_flags) if ~present]
# 	present_exact_match_scores = valid_match_scores_exact[valid_present_pred_flags]
# 	absent_exact_match_scores = valid_match_scores_exact[~valid_present_pred_flags]
#
# 	all_exact_results = run_classic_metrics(valid_match_scores_exact, valid_preds, tgt_seqs, metric_names, topk_range)
# 	present_exact_results = run_classic_metrics(present_exact_match_scores, valid_present_preds, present_tgts, metric_names, topk_range)
# 	absent_exact_results = run_classic_metrics(absent_exact_match_scores, valid_absent_preds, absent_tgts, metric_names, absent_topk_range)
#
# 	eval_results_names = ['all_exact', 'present_exact', 'absent_exact']
# 	eval_results_list = [all_exact_results, present_exact_results, absent_exact_results]
#
# 	print_out = print_predeval_result(src_text,
# 									  tgt_seqs, present_tgt_flags,
# 									  pred_seqs, pred_scores, present_pred_flags, valid_pred_flags,
# 									  valid_and_present_flags, valid_and_absent_flags, match_scores_exact,
# 									  eval_results_names, eval_results_list)
#
# 	if return_eval:
# 		eval_results_dict = {
# 			'all_exact': all_exact_results,
# 			'present_exact': present_exact_results,
# 			'absent_exact': absent_exact_results
# 		}
# 		return print_out, eval_results_dict
# 	else:
# 		return print_out


# def print_predeval_result(src_text,
# 						  tgt_seqs, present_tgt_flags,
# 						  pred_seqs, pred_scores, present_pred_flags, valid_pred_flags,
# 						  valid_and_present_flags, valid_and_absent_flags, match_scores_exact,
# 						  results_names, results_list):
# 	print_out = '=' * 50
# 	print_out += '\n[Source]: %s \n' % (src_text)
#
# 	print_out += '[GROUND-TRUTH] #(all)=%d, #(present)=%d, #(absent)=%d\n' % \
# 				 (len(present_tgt_flags), sum(present_tgt_flags), len(present_tgt_flags)-sum(present_tgt_flags))
# 	print_out += '\n'.join(
# 		['\t\t[%s]' % ' '.join(phrase) if is_present else '\t\t%s' % ' '.join(phrase) for phrase, is_present in
# 		 zip(tgt_seqs, present_tgt_flags)])
#
# 	print_out += '\n[PREDICTION] #(all)=%d, #(valid)=%d, #(present)=%d, ' \
# 				 '#(valid&present)=%d, #(valid&absent)=%d\n' % (
# 		len(pred_seqs), sum(valid_pred_flags), sum(present_pred_flags),
# 		sum(valid_and_present_flags), sum(valid_and_absent_flags))
# 	print_out += ''
# 	preds_out = ''
# 	for p_id, (word, match, is_valid, is_present) in enumerate(
# 		zip(pred_seqs, match_scores_exact, valid_pred_flags, present_pred_flags)):
# 		score = pred_scores[p_id] if pred_scores else "Score N/A"
#
# 		preds_out += '%s\n' % (' '.join(word))
# 		if is_present:
# 			print_phrase = '[%s]' % ' '.join(word)
# 		else:
# 			print_phrase = ' '.join(word)
#
# 		if match == 1.0:
# 			correct_str = '[correct!]'
# 		else:
# 			correct_str = ''
#
# 		pred_str = '\t\t%s\t%s \t%s\n' % ('[%.4f]' % (-score) if pred_scores else "Score N/A",
# 												print_phrase, correct_str)
# 		if not is_valid:
# 			pred_str = '\t%s' % pred_str
#
# 		print_out += pred_str
#
# 	print_out += "\n ======================================================= \n"
#
# 	print_out += '[GROUND-TRUTH] #(all)=%d, #(present)=%d, #(absent)=%d\n' % \
# 				 (len(present_tgt_flags), sum(present_tgt_flags), len(present_tgt_flags)-sum(present_tgt_flags))
# 	print_out += '\n[PREDICTION] #(all)=%d, #(valid)=%d, #(present)=%d, ' \
# 				 '#(valid&present)=%d, #(valid&absent)=%d\n' % (
# 		len(pred_seqs), sum(valid_pred_flags), sum(present_pred_flags),
# 		sum(valid_and_present_flags), sum(valid_and_absent_flags))
#
# 	for name, results in zip(results_names, results_list):
# 		# print @5@10@O@M for present_exact, print @50@M for absent_exact
# 		if name in ['all_exact', 'present_exact', 'absent_exact']:
# 			if name.startswith('all') or name.startswith('present'):
# 				topk_list = ['k', '10']
# 			else:
# 				topk_list = ['50', 'M']
#
# 			for topk in topk_list:
# 				print_out += "\n --- batch {} F1 @{}: \t".format(name, topk) \
# 							 + "{:.4f}".format(results['f_score@{}'.format(topk)])
# 		else:
# 			# ignore partial for now
# 			continue
#
# 	print_out += "\n ======================================================="
#
# 	return print_out
def apply_set_on_kps_at_macro_level(sequence_generated_kps:str) :
	"""
	Apply a set over generated KPs in order to merge duplicate KPs
	:param sequence_generated_kps:
	:return:
	"""
	sequence_generated_kps = list(set(sequence_generated_kps.split(KPS_SEPARATOR)))
	sequence_generated_kps = [kp.strip() for kp in sequence_generated_kps]
	return KPS_SEPARATOR.join(sequence_generated_kps)

def filter_stuttered_kp(sequence_generated_kps:str) :
	"""
	For KPs containing repeated patterns, reduce stuttering
	Then merge duplicate KPs
	:param sequence_generated_kps:
	:return:
	"""
	new_sequence_kps = []
	for kp in sequence_generated_kps.split(KPS_SEPARATOR) :
		kp = kp.strip()
		# skip KPs with only 1 character
		if len(kp) == 1 or len(kp) == 0 :
			continue
		#
		# if less than 5 distinct characters and the most frequent characters occurs more than 60 times
		elif len(Counter(kp)) < 5 and Counter(kp).most_common()[0][1] > 100 :
			# skip if the kp is a long succession uninterrupted characters
			if len(kp.split()) == 1 :
				print(f"Skipped: {kp}")
				continue
			# kp has several elements separated by whitespaces
			# merge then add
			else :
				kp = " ".join(list(set(kp.split())))
				print(f"Reduced: {kp}")
				new_sequence_kps.append(kp)
		#
		# kp should not have too much stuttering
		else :
			new_sequence_kps.append(kp)
	return KPS_SEPARATOR.join(new_sequence_kps)


def remove_stuttered_kp(sequence_generated_kps:str) :
	"""
	For KPs containing repeated patterns, REMOVE stuttering
	Then merge duplicate KPs
	:param sequence_generated_kps:
	:return:
	"""
	new_sequence_kps = []
	for kp in sequence_generated_kps.split(KPS_SEPARATOR) :
		kp = kp.strip()
		# skip KPs with only 1 character
		if len(kp) == 1 or len(kp) == 0 :
			continue
		#
		# if less than 5 distinct characters and the most frequent characters occurs more than 60 times
		elif len(Counter(kp)) < 5 and Counter(kp).most_common()[0][1] > 100 :
			# skip if the kp is a long succession uninterrupted characters
			if len(kp.split()) == 1 :
				print(f"Skipped: {kp}")
				continue
			# kp has several elements separated by whitespaces
			# merge then add
			else :
				#kp = " ".join(list(set(kp.split())))
				print(f"Removed: {kp}")
				#new_sequence_kps.append(kp)
		#
		# kp should not have too much stuttering
		else :
			new_sequence_kps.append(kp)
	return KPS_SEPARATOR.join(new_sequence_kps)


def apply_nltk_stemming(
		text:str,
		stemmer
) :
	"""
	Function used in recompute_scores_on_generated_kps for apply some nltk stemmer on a string (keyphrase or text)
	:param text:
	:param stemmer:
	:return:
	"""
	text = [stemmer.stem(token) for token in str(text).split()] # text may sometimes be a float, hence str()
	return str(' '.join(text))

def apply_cognate_stemming(text: str,
						   lang: str='et') -> str:
	text = [token[:4] if len(token) >= 4 else token for token in str(text).split()]
	return str(' '.join(text))

stemmer_d = {
		'en': PorterStemmer().stem,
		'da': SnowballStemmer(language='danish').stem,
		'nl': SnowballStemmer(language='dutch').stem,
		'fi': SnowballStemmer(language='finnish').stem,
		'fr': SnowballStemmer(language='french').stem,
		'de': SnowballStemmer(language='german').stem,
		'hu': SnowballStemmer(language='hungarian').stem,
		'it': SnowballStemmer(language='italian').stem,
		'pt': SnowballStemmer(language='portuguese').stem,
		'ro': SnowballStemmer(language='romanian').stem,
		'es': SnowballStemmer(language='spanish').stem,
		'sv': SnowballStemmer(language='swedish').stem,
		'el': GreekStemmer().stem,
		'hr': CroatianStemmer().stem,
		'mt': MalteseStemmer().stem,
		'cs': cz_stem,
		'lt': extra_sbl_stemmer('lt'),
		'ga': extra_sbl_stemmer('ga'),
		'lv': LatvianStemmer.stem,
		'sl': MirkoStemmer().mirko_pipeline,
		'pl': StempelStemmer.polimorf().stem,
		'sk': sk_stem,
		'bg': BulStemmer.from_file('stem-context-1', min_freq=2, left_context=1).stem,
		'et': apply_cognate_stemming, #estonian_shell_stemmer,
	}

def apply_stemming(text: str, lang: str) -> str:
	"""
	Function used in recompute_scores_on_generated_kps for apply some nltk stemmer on a string (keyphrase or text)
	"""
	text = [stemmer_d[lang](token) for token in str(text).split()] # text may sometimes be a float, hence str()
	text = [token for token in text if token is not None]
	return str(' '.join(text))




translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
def apply_custom_nomalization(
		text:str,
		is_kp_sequence:bool,
		stemmer,
		list_stopwords:List[str]=None,
) :
	text = str(text).lower()

	if stemmer is not None : # apply stemming if stemmer available
		text = ' '.join([stemmer.stem(token) for token in text.split()])  # text may sometimes be a float, hence str()

	text = text.replace('°', 'o ') # n.° becomes n.o
	text = re.sub(r"\(.*?\)", ' ', text) # remove parenthesis 'Article 23(1)(e) and (j)' -> 'Article 23   and  '

	# replace punctuation with whitespace
	punctuation = '!"#$%&\'()*+,-./:<=>?@[\\]^_`{|}~' # '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~' # take away ; from punctuation
	translator = str.maketrans(punctuation, ' ' * len(punctuation))  # map punctuation to space
	text = ' ; '.join([kp.translate(translator) for kp in text.split(' ; ')])

	if list_stopwords is not None :
		text = ' '.join([token for token in text.split() if token not in list_stopwords])

	text = re.sub(r" +", " ", text)
	return text.strip()


dict_stemmer = {
	'en': apply_stemming,
	'da': apply_stemming,
	'nl': apply_stemming,
	'fi': apply_stemming,
	'fr': apply_stemming,
	'de': apply_stemming,
	'hu': apply_stemming,
	'it': apply_stemming,
	'pt': apply_stemming,
	'ro': apply_stemming,
	'es': apply_stemming,
	'sv': apply_stemming,

	'el': apply_stemming,
	'lt': apply_stemming,
	'et': apply_cognate_stemming,
	'cs': apply_stemming,
	'lv': apply_stemming,
	'sl': apply_stemming,
	'pl': apply_stemming,
	'mt': apply_stemming,
	'sk': apply_stemming,
	'bg': apply_stemming,
	'hr': apply_stemming,
	'ga': apply_stemming,
}

dict_stopwords = {
	'da': stopwords.words('danish'),
	'nl': stopwords.words('dutch'),
	'en': stopwords.words('english'),
	'fi': stopwords.words('finnish'),
	'fr': stopwords.words('french'),

	'de': stopwords.words('german'),
	'el': stopwords.words('greek'),
	'hu': stopwords.words('hungarian'),
	'it': stopwords.words('italian'),
	'pt': stopwords.words('portuguese'),

	'ro': stopwords.words('romanian'),
	'sl': stopwords.words('slovene'),
	'es': stopwords.words('spanish'),
	'sv': stopwords.words('swedish'),
	# ['arabic',
	# 'azerbaijani',
	# 'indonesian',
	# 'kazakh',
	# 'nepali',
	# 'norwegian',
	# 'russian',
	# 'tajik',
	# 'turkish']
}

def recompute_scores_on_generated_kps(
        path_to_folder_with_output_kps:str,
        output_kps_filename:str,
        output_filename_prefix:str='computed',
		toy:int=1,
		preprocessing:Literal[
			'no_preprocessing',
			'no_preprocessing_stemming',
			'merge_duplicate_kp',
			'filter_stutter',
			'remove_all_stuttering',
			'classic_en_stemming',
			'stemming_mu',
			'custom_normalization',
		]='no_preprocessing',
) :
	"""
    28-09-2023
    This method can be run directly from the command line
    It recomputes scores on generated KPs for sanity check
    :param path_to_folder_with_output_kps: path to folder with generated KPs, must end with '/' (e.g. /data/rali6/Tmp/salaunol/_NEXT/a22/chantier3_curia/eurlex_expansion/instances_kw_decisions/split_2011_2016/FullEuropaMu/output/full_mbart50/2023-09-22_13-18-26_customLexKey_len1k_linearDecay_1e-4_bs16_beams10/)
    :param output_kps_filename: name of TSV with generated KPs (e.g. target_generated_kps.tsv)
    :param output_filename_prefix: prefix added to new files with scores
    :param preprocessing: choice of preprocessing
    :param toy: 1 for a quick run of the pipeline with 100 instances, 0 for an evaluation with all the test instances (may last 1h20 minutes)
    :return: nothing, print overall scores and save two TSV to disk in path_to_folder
    """
	start = time.perf_counter()
	if toy == 0 :
		nrows = None # set to 100 for quick test of evaluation pipeline else use None
	elif toy == 1 :
		nrows = 100
	#part = 'toy' if toy else 'full'
	#PATH_TO_EXPECTED_TARGET = f"/data/rali6/Tmp/salaunol/_NEXT/a22/chantier3_curia/eurlex_expansion/instances_kw_decisions/split_2011_2016/{part.title()}EuropaMu/" # test set from chronological split
	#PATH_TO_EXPECTED_TARGET = f"/data/rali6/Tmp/salaunol/_NEXT/a22/chantier3_curia/eurlex_expansion/instances_kw_decisions/random_split/RandomSplit_FullEuropaMu/" # uncomment if need test from the random split
	#print(f"\n> Loading target KPs from {PATH_TO_EXPECTED_TARGET}...")
	print(f"\n>> Loading target KPs and input text from Europa test set on huggingface...")
	print(f"/!\\ This step may take a while!\n")
	# cols = "lang_celex_id input_text language".split()
	#df_test = pd.read_csv(PATH_TO_EXPECTED_TARGET + 'test.tsv',
	#					  sep='\t',
	#					  low_memory=False,
	#					  usecols=cols,
	#					  nrows=nrows)
	df_test = load_dataset("NCube/europa", split="test")
	df_test = pd.DataFrame(df_test)
	if nrows is not None :
		df_test = df_test.head(nrows)

	df_test = df_test.rename(columns={'lang' : 'language', 'keyphrases' : 'target_keyphrases'}) # rename some columns
	df_test['lang_celex_id'] = df_test['language'] + df_test['celex_id'] # celex_id can be the same for different translations for the same judgment, lang_celex_id is a unique identifier
	#print(df_test.shape)
	#print(df_test.tail(5))


	# apply preprocessing on input text
	if preprocessing == 'classic_en_stemming' :
		df_test = df_test[df_test.language == 'en'].reset_index(drop=True)
	elif preprocessing == 'stemming_mu' :
		print('> Apply stemming for all 24 languages on input_text...')
		print('/!\\ This step may take around or over an hour!\n')
		print('BEFORE stemming:')
		print(df_test.tail(5))
		col_input_text = df_test.input_text.tolist()
		for row_idx in tqdm.tqdm(range(df_test.shape[0])) :
			lang = df_test.loc[row_idx, 'language']
			if lang in dict_stemmer :
				col_input_text[row_idx] = apply_stemming(col_input_text[row_idx], lang)
		df_test['input_text'] = col_input_text
		print('AFTER stemming:')

	elif preprocessing == 'custom_normalization' :
		print('> Apply customized normalization')
		print('BEFORE normalization:')
		print(df_test.tail(5))
		col_input_text = df_test.input_text.tolist()
		for row_idx in tqdm.tqdm(range(df_test.shape[0])) :
			lang = df_test.loc[row_idx, 'language']
			col_input_text[row_idx] = apply_custom_nomalization(
				text=col_input_text[row_idx],
				is_kp_sequence=False,
				stemmer=None if lang not in dict_stemmer else dict_stemmer[lang],
				list_stopwords=None if lang not in dict_stopwords else dict_stopwords[lang],
			)
		df_test['input_text'] = col_input_text
		print('AFTER normalization:')
	else :
		raise Exception(f"Choice of preprocessing '{preprocessing}' could not be found.")

	del df_test['language']
	print(df_test.tail(5))
	print(f"df_test.shape {df_test.shape}\n")

	print(f"\n> Loading generated KPs from {path_to_folder_with_output_kps + output_kps_filename}...")
	#cols = "language celex_id target_keyphrases generated_keyphrases".split()
	cols = "language celex_id generated_keyphrases".split()
	df_candidates = pd.read_csv(path_to_folder_with_output_kps + output_kps_filename,
								sep='\t',
								low_memory=False,
								usecols=cols,
								nrows=nrows)
	df_candidates['lang_celex_id'] = df_candidates['language'] + df_candidates['celex_id']

	# transfer target_keyphrases column from df_test to df_candidates
	col_target_kps = df_test.target_keyphrases.tolist()
	df_test["target_keyphrases"] = [KPS_SEPARATOR.join(sequence_kps) for sequence_kps in col_target_kps]
	df_candidates = df_candidates.merge(df_test[["lang_celex_id", "target_keyphrases"]], how='left', on='lang_celex_id')
	del df_test["target_keyphrases"]


	if preprocessing == 'classic_en_stemming' :
		df_candidates = df_candidates[df_candidates.language == 'en'].reset_index(drop=True)
	print(df_candidates.tail(2))
	print(f"df_candidates.shape {df_candidates.shape}\n")

	if preprocessing in ['no_preprocessing', 'classic_en_stemming'] :
		print("Candidates KPs are not modified.")
		col_generated_kps = df_candidates.generated_keyphrases.tolist()
		col_generated_kps = [str(e) for e in col_generated_kps] # some candidates are values that can be considered as float, we make sure that they are considered as str
		df_candidates['generated_keyphrases'] = col_generated_kps

	elif preprocessing == 'merge_duplicate_kp' :
		print("Apply set on all KPs in order to reduce duplicates")
		col_generated_kps = df_candidates.generated_keyphrases.tolist()
		col_generated_kps = [apply_set_on_kps_at_macro_level(e) for e in tqdm.tqdm(col_generated_kps)]
		df_candidates['generated_keyphrases'] = col_generated_kps

	elif preprocessing == 'filter_stutter' :
		print("Filter stuttered KPs") # keep reduced form of stuttering
		col_generated_kps = df_candidates.generated_keyphrases.tolist()
		col_generated_kps = [filter_stuttered_kp(e) for e in tqdm.tqdm(col_generated_kps)]
		df_candidates['generated_keyphrases'] = col_generated_kps

	elif preprocessing == 'remove_all_stuttering' :
		print("Remove all stuttering") # remove all stuttering
		col_generated_kps = df_candidates.generated_keyphrases.tolist()
		col_generated_kps = [remove_stuttered_kp(e) for e in tqdm.tqdm(col_generated_kps)]
		df_candidates['generated_keyphrases'] = col_generated_kps

	elif preprocessing == 'stemming_mu' :
		print('> Apply stemming on target KPs and generated KPs...')
		print('BEFORE stemming:')
		print(df_candidates.tail(5))
		col_generated_kps = df_candidates.generated_keyphrases.tolist()
		col_target_kps = df_candidates.target_keyphrases.tolist()

		for row_idx in tqdm.tqdm(range(df_candidates.shape[0])) :
			lang = df_candidates.loc[row_idx, 'language']
			if lang in dict_stemmer :

				gen_kp = apply_stemming(col_generated_kps[row_idx], lang), #dict_stemmer[lang])
				tar_kp = apply_stemming(col_target_kps[row_idx], lang), #dict_stemmer[lang])
				col_generated_kps[row_idx] = gen_kp[0] if type(gen_kp) is tuple else gen_kp
				col_target_kps[row_idx] = tar_kp[0] if type(tar_kp) is tuple else tar_kp
		df_candidates['generated_keyphrases'] = col_generated_kps
		df_candidates['target_keyphrases'] = col_target_kps
		print('AFTER stemming:')
		print(df_candidates.tail(5))
	elif preprocessing == 'custom_normalization' :
		print('> Apply normalization...')
		print('BEFORE normalization:')
		print(df_candidates.tail(5))
		col_generated_kps = df_candidates.generated_keyphrases.tolist()
		col_target_kps = df_candidates.target_keyphrases.tolist()

		for row_idx in tqdm.tqdm(range(df_candidates.shape[0])) :
			lang = df_candidates.loc[row_idx, 'language']

			col_generated_kps[row_idx] = apply_custom_nomalization(
				text=col_generated_kps[row_idx],
				is_kp_sequence=True,
				stemmer=None if lang not in dict_stemmer else dict_stemmer[lang],
				list_stopwords=None if lang not in dict_stopwords else dict_stopwords[lang],
			)
			col_target_kps[row_idx] = apply_custom_nomalization(
				text=col_target_kps[row_idx],
				is_kp_sequence=True,
				stemmer=None if lang not in dict_stemmer else dict_stemmer[lang],
				list_stopwords=None if lang not in dict_stopwords else dict_stopwords[lang],
			)

		df_candidates['generated_keyphrases'] = col_generated_kps
		df_candidates['target_keyphrases'] = col_target_kps
		print('AFTER normalization:')
		print(df_candidates.tail(5))
	else :
		raise Exception(f"Choice of preprocessing '{preprocessing}' could not be found.")


	affix_preprocessing = ' '.join(preprocessing.split('_')).title().replace(' ', '')

	print("\n> Merge df_test and df_candidates along column lang_celex_id")
	df_candidates = df_candidates.merge(df_test, how='inner', on='lang_celex_id')
	print(df_candidates.tail(2))
	print(f"df_candidates.shape {df_candidates.shape}\n")
	if preprocessing == 'stemming_mu':
		print('Compute present/absent KPs stats with stemming applied on all 24 EU languages')
		col_ratio_present_kps = []
		col_ratio_absent_kps = []
		col_nbr_kps = []
		for row_idx in tqdm.tqdm(range(df_candidates.shape[0])) :
			list_present_kps = []
			list_absent_kps = []
			kps = df_candidates.loc[row_idx, 'target_keyphrases'].split(' ; ')
			col_nbr_kps.append(len(kps))
			for kp in kps :
				kp = str(kp)
				if kp.lower() in str(df_candidates.loc[row_idx, 'input_text']).lower() :
					list_present_kps.append(kp)
				else :
					list_absent_kps.append(kp)
			col_ratio_present_kps.append(len(list_present_kps) / len(kps))
			col_ratio_absent_kps.append(len(list_absent_kps) / len(kps))
		df_candidates['nbr_kps'] = col_nbr_kps
		df_candidates['ratio_present_kps'] = col_ratio_present_kps
		df_candidates['ratio_absent_kps'] = col_ratio_absent_kps
		print(df_candidates.ratio_present_kps.describe().round(3))
		print(df_candidates.ratio_absent_kps.describe().round(3))
		langs = ['fr', 'de', 'en', 'it', 'nl', 'el',
				 'da', 'pt', 'es', 'sv', 'fi', 'lt',
				 'et', 'cs', 'hu', 'lv', 'sl', 'pl',
				 'mt', 'sk', 'ro', 'bg', 'hr', 'ga']
		print(df_candidates.groupby('language').ratio_present_kps.mean().round(3).reindex(langs))
		print(df_candidates.groupby('language').ratio_absent_kps.mean().round(3).reindex(langs))

	print("_" * 100)
	print("\n> Computing individual scores...")
	if preprocessing == 'classic_en_stemming' :
		scores = evaluate(
			src_list=df_candidates.input_text.tolist(),
			tgt_list=df_candidates.target_keyphrases.str.split(" ; ").tolist(),
			pred_list=df_candidates.generated_keyphrases.str.split(" ; ").tolist(),
			language='en',
			tokenizer='spacy',
			unk_token="<unk>",
		)
	else :
		scores = evaluate(
			src_list=df_candidates.input_text.tolist(),
			tgt_list=df_candidates.target_keyphrases.str.split(" ; ").tolist(),
			pred_list=df_candidates.generated_keyphrases.str.split(" ; ").tolist(),
			language='mu',
			tokenizer='split_nopunc',
			unk_token="<unk>",
		)

	for k in scores :
		df_candidates[k] = scores[k]

	recomputed_scores_filename = f"{path_to_folder_with_output_kps}{output_filename_prefix}_preprocess{affix_preprocessing}_individualScores_{output_kps_filename}"
	print(f"\n\n>> Saving new TSV with recomputed INDIVIDUAL scores at: {recomputed_scores_filename}")
	del df_candidates['input_text']
	df_candidates.to_csv(recomputed_scores_filename, sep='\t', index=False)

	#########################################
	print("_"*100)
	print("\n> Compute stats for all languages and each language (focus on hard version)")

	list_rows = []


	# compute micro average (average scores over all instances, high-resource languages can pull up the 'all_micro' scores)
	row_dict = {
		'language' : 'weighted_avg',# 'all_micro',
		#'f@5': round(df_candidates['all_exact_f_score_hard@5'].mean(), 4),
		#'f@M' : round(df_candidates['all_exact_f_score_hard@M'].mean(), 4),

		# the 4 metrics reported in the paper
		#'f@5_present' : round(df_candidates['present_exact_f_score_hard@5'].mean(), 4),
		#'f@M_present' : round(df_candidates['present_exact_f_score_hard@M'].mean(), 4),
		#'f@5_absent' : round(df_candidates['absent_exact_f_score_hard@5'].mean(), 4),
		#'f@M_absent' : round(df_candidates['absent_exact_f_score_hard@M'].mean(), 4),

		# pour relecteur retardataire #  (F1@10, MAP@50, P@10, R@10, F@10)
		'f@5_present': round(df_candidates['present_exact_f_score_hard@5'].mean(),4),

		'p@10_present_nh': round(df_candidates['present_exact_precision@10'].mean(), 4),  #
		'p@10_present_h': round(df_candidates['present_exact_precision_hard@10'].mean(), 4),  #
		'r@10_present': round(df_candidates['present_exact_recall@10'].mean(), 4), #
		'f@10_present': round(df_candidates['present_exact_f_score_hard@10'].mean(), 4),  #

		'f@15_present': round(df_candidates['present_exact_f_score_hard@15'].mean(), 4),
		'f@k_present': round(df_candidates['present_exact_f_score_hard@k'].mean(), 4),
		'f@M_present': round(df_candidates['present_exact_f_score_hard@M'].mean(), 4),
		'f@1_present': round(df_candidates['present_exact_f_score_hard@1'].mean(), 4),
		'f@3_present': round(df_candidates['present_exact_f_score_hard@3'].mean(), 4),

		'f@5_absent': round(df_candidates['absent_exact_f_score_hard@5'].mean(), 4),

		'p@10_absent_nh': round(df_candidates['absent_exact_precision@10'].mean(), 4),  # no 'hard' recall
		'p@10_absent_h': round(df_candidates['absent_exact_precision_hard@10'].mean(), 4), # no 'hard' recall
		'r@10_absent': round(df_candidates['absent_exact_recall@10'].mean(), 4), # no 'hard' recall
		'f@10_absent': round(df_candidates['absent_exact_f_score_hard@10'].mean(), 4),

		'f@15_absent': round(df_candidates['absent_exact_f_score_hard@15'].mean(), 4),
		'f@50_absent': round(df_candidates['absent_exact_f_score_hard@50'].mean(), 4),
		'f@k_absent': round(df_candidates['absent_exact_f_score_hard@k'].mean(), 4),
		'f@M_absent': round(df_candidates['absent_exact_f_score_hard@M'].mean(), 4),
		'map@50' : round(df_candidates['all_exact_MAP@50'].mean(), 4),

	}

	list_rows.append(row_dict)


	# compute average scores for each language
	list_languages_rows = []
	for language in ['fr', 'de', 'en', 'it', 'nl', 'el',
					 'da', 'pt', 'es', 'sv', 'fi', 'lt',
					 'et', 'cs', 'hu', 'lv', 'sl', 'pl',
					 'mt', 'sk', 'ro', 'bg', 'hr', 'ga'] :
		if language in df_candidates.language.unique() :
			row_dict = {
				'language': language,
				#'f@5': round(df_candidates[df_candidates.language == language]['all_exact_f_score_hard@5'].mean(), 4),
				#'f@M': round(df_candidates[df_candidates.language == language]['all_exact_f_score_hard@M'].mean(), 4),

				# à conserver
				#'f@5_present': round(df_candidates[df_candidates.language == language]['present_exact_f_score_hard@5'].mean(), 4),
				#'f@M_present': round(df_candidates[df_candidates.language == language]['present_exact_f_score_hard@M'].mean(), 4),
				#'f@5_absent': round(df_candidates[df_candidates.language == language]['absent_exact_f_score_hard@5'].mean(), 4),
				#'f@M_absent': round(df_candidates[df_candidates.language == language]['absent_exact_f_score_hard@M'].mean(), 4),

				# pour relecteur retardataire #  (F1@10, MAP@50, P@10, R@10, F@10)
				'f@5_present': round(df_candidates[df_candidates.language == language]['present_exact_f_score_hard@5'].mean(), 4),

				'p@10_present_nh' : round(df_candidates[df_candidates.language == language]['present_exact_precision@10'].mean(), 4),     #
				'p@10_present_h' : round(df_candidates[df_candidates.language == language]['present_exact_precision_hard@10'].mean(), 4), #
				'r@10_present' : round(df_candidates[df_candidates.language == language]['present_exact_recall@10'].mean(), 4),           #
				'f@10_present': round(df_candidates[df_candidates.language == language]['present_exact_f_score_hard@10'].mean(), 4),      #

				'f@15_present': round(df_candidates[df_candidates.language == language]['present_exact_f_score_hard@15'].mean(), 4),
				'f@k_present': round(df_candidates[df_candidates.language == language]['present_exact_f_score_hard@k'].mean(), 4),
				'f@M_present': round(df_candidates[df_candidates.language == language]['present_exact_f_score_hard@M'].mean(), 4),
				'f@1_present': round(df_candidates[df_candidates.language == language]['present_exact_f_score_hard@1'].mean(), 4),
				'f@3_present': round(df_candidates[df_candidates.language == language]['present_exact_f_score_hard@3'].mean(), 4),

				'f@5_absent': round(df_candidates[df_candidates.language == language]['absent_exact_f_score_hard@5'].mean(), 4),

				'p@10_absent_nh': round(df_candidates[df_candidates.language == language]['absent_exact_precision@10'].mean(), 4),           # no 'hard' recall
				'p@10_absent_h': round(df_candidates[df_candidates.language == language]['absent_exact_precision_hard@10'].mean(), 4), # no 'hard' recall
				'r@10_absent': round(df_candidates[df_candidates.language == language]['absent_exact_recall@10'].mean(), 4),  # no 'hard' recall
				'f@10_absent': round(df_candidates[df_candidates.language == language]['absent_exact_f_score_hard@10'].mean(), 4),

				'f@15_absent': round(df_candidates[df_candidates.language == language]['absent_exact_f_score_hard@15'].mean(), 4),
				'f@50_absent': round(df_candidates[df_candidates.language == language]['absent_exact_f_score_hard@50'].mean(), 4),
				'f@k_absent': round(df_candidates[df_candidates.language == language]['absent_exact_f_score_hard@k'].mean(), 4),
				'f@M_absent': round(df_candidates[df_candidates.language == language]['absent_exact_f_score_hard@M'].mean(), 4),
				'map@50': round(df_candidates[df_candidates.language == language]['all_exact_MAP@50'].mean(), 4),
			}
			list_languages_rows.append(row_dict)


	# compute unweighted average (each language has the same weight in the 'all_macro' scores)
	df_avg_scores_per_lang = df_candidates.groupby('language').mean(numeric_only=True)
	row_dict = {
		'language' : 'unweighted_avg', # 'all_macro',
		#'f@5': round(df_candidates['all_exact_f_score_hard@5'].mean(), 4),
		#'f@M' : round(df_candidates['all_exact_f_score_hard@M'].mean(), 4),

		# the 4 reported metrics in the paper
		#'f@5_present' : round(df_avg_scores_per_lang['present_exact_f_score_hard@5'].mean(), 4),
		#'f@M_present' : round(df_avg_scores_per_lang['present_exact_f_score_hard@M'].mean(), 4),
		#'f@5_absent' : round(df_avg_scores_per_lang['absent_exact_f_score_hard@5'].mean(), 4),
		#'f@M_absent' : round(df_avg_scores_per_lang['absent_exact_f_score_hard@M'].mean(), 4),

		# pour relecteur retardataire #  (F1@10, MAP@50, P@10, R@10, F@10)
		'f@5_present': round(df_avg_scores_per_lang['present_exact_f_score_hard@5'].mean(), 4),

		'p@10_present_nh': round(df_avg_scores_per_lang['present_exact_precision@10'].mean(), 4),  #
		'p@10_present_h': round(df_avg_scores_per_lang['present_exact_precision_hard@10'].mean(), 4),  #
		'r@10_present': round(df_avg_scores_per_lang['present_exact_recall@10'].mean(), 4),  #
		'f@10_present': round(df_avg_scores_per_lang['present_exact_f_score_hard@10'].mean(), 4),  #

		'f@15_present': round(df_avg_scores_per_lang['present_exact_f_score_hard@15'].mean(), 4),
		'f@k_present': round(df_avg_scores_per_lang['present_exact_f_score_hard@k'].mean(), 4),
		'f@M_present': round(df_avg_scores_per_lang['present_exact_f_score_hard@M'].mean(), 4),
		'f@1_present': round(df_avg_scores_per_lang['present_exact_f_score_hard@1'].mean(), 4),
		'f@3_present': round(df_avg_scores_per_lang['present_exact_f_score_hard@3'].mean(), 4),

		'f@5_absent': round(df_avg_scores_per_lang['absent_exact_f_score_hard@5'].mean(), 4),

		'p@10_absent_nh': round(df_avg_scores_per_lang['absent_exact_precision@10'].mean(), 4),  # no 'hard' recall
		'p@10_absent_h': round(df_avg_scores_per_lang['absent_exact_precision_hard@10'].mean(), 4),  # no 'hard' recall
		'r@10_absent': round(df_avg_scores_per_lang['absent_exact_recall@10'].mean(), 4),  # no 'hard' recall
		'f@10_absent': round(df_avg_scores_per_lang['absent_exact_f_score_hard@10'].mean(), 4),

		'f@15_absent': round(df_avg_scores_per_lang['absent_exact_f_score_hard@15'].mean(), 4),
		'f@50_absent': round(df_avg_scores_per_lang['absent_exact_f_score_hard@50'].mean(), 4),
		'f@k_absent': round(df_avg_scores_per_lang['absent_exact_f_score_hard@k'].mean(), 4),
		'f@M_absent': round(df_avg_scores_per_lang['absent_exact_f_score_hard@M'].mean(), 4),
		'map@50' : round(df_avg_scores_per_lang['all_exact_MAP@50'].mean(), 4),


	}
	list_rows.append(row_dict)
	list_rows += list_languages_rows

	df_summary_scores = pd.DataFrame(list_rows)
	recomputed_scores_filename = f"{path_to_folder_with_output_kps}{output_filename_prefix}_preprocess{affix_preprocessing}_languageScores_{output_kps_filename}"
	print(df_summary_scores)
	print(f"\n\n>> Saving new TSV with recomputed LANGUAGES scores at: {recomputed_scores_filename}")
	df_summary_scores.to_csv(recomputed_scores_filename, sep='\t', index=False)

	time_taken = time.perf_counter() - start
	print('Computing time:', datetime.timedelta(seconds=time_taken))