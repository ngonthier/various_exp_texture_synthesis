#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on August 1
Style Transfer main

@author: nicolas
"""
import Style_Transfer as st
from Arg_Parser import get_parser_args 

def main():
	#global args
	try:
		parser = get_parser_args()
		args = parser.parse_args()
		st.style_transfer(args)
	except:
		raise

if __name__ == '__main__':
	main()
