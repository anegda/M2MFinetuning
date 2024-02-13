#!/usr/bin/env python

from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

from m2m_multiling_tune import loadtok

import sys
import re
import torch
import os

from datetime import datetime

def log(msg):
	print(str(datetime.now()) + ": " + msg, file = sys.stderr)

def tokenize(tok, snt, srcLang):
	tok.src_lang = srcLang
	return tok(snt, return_tensors="pt", padding=True).to("cuda")
	#return tok(snt, return_tensors="pt", padding=True)

def translate(mdl, tok, enctxt, trgLang):
	generated_tokens = mdl.generate(**enctxt, forced_bos_token_id=tok.get_lang_id(trgLang))

	return tok.batch_decode(generated_tokens, skip_special_tokens=True)

def loadmdl(mdl):
	log("Load model")
	model = M2M100ForConditionalGeneration.from_pretrained(mdl)
	
	log("Model to GPU")
	model.to("cuda")
	return model

def fixit(self):
	self.id_to_lang_token = dict(list(self.id_to_lang_token.items()) + list(self.added_tokens_decoder.items()))
	self.lang_token_to_id = dict(list(self.lang_token_to_id.items()) + list(self.added_tokens_encoder.items()))
	self.lang_code_to_token = { k.replace("_", ""): k for k in self.additional_special_tokens }
	self.lang_code_to_id = { k.replace("_", ""): v for k, v in self.lang_token_to_id.items() }

def m2mTranslate(srcList, srcLang, tgtLang):
	enc = tokenize(tokenizer, srcList, srcLang)
	out = translate(model, tokenizer, enc, tgtLang)
	return out

if __name__ == "__main__":
	mdlname = sys.argv[1]
	
	log("Loading tokenizer")
	tokenizer = loadtok(mdlname)
	
	model = loadmdl(mdlname)

	lp = sys.argv[2]

	(srcLang, tgtLang) = lp.split("-")

	input_file_path = 'input.eu'  # Replace with the path to your input file
	output_file_path = 'output.MTen'  # Replace with the desired output file path

	with open(input_file_path, "r", encoding="utf8") as input_file:
		with open(output_file_path, "w+", encoding="utf8") as output_file:
			line: str = input_file.readline()
			while line:
				translation = m2mTranslate(line, srcLang, tgtLang)
				print(translation[0], file=output_file)
				line = input_file.readline()