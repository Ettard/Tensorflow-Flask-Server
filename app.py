#!flask/bin/python
import urllib
import sys
import io
import os
from PIL import Image

from predict import Predictor
from flask import Flask, jsonify, request

app = Flask(__name__)
list_predictor = []
dic_ckpt_name = {}
	
@app.route('/flask/api/v1.0/tasks', methods=['GET'])
def get_tasks():
	try:
		list_result = []
		url = urllib.unquote(request.args.get('img_url'))
		data = urllib.urlopen(url).read()
		img = Image.open(io.BytesIO(data))
		
		for pred in list_predictor:
			result = pred.get_predict(img)
		list_result.extend(result)
		return jsonify(res_dic)		
	except Exception,e:
		print e
		return '{}'

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'})

if __name__ == '__main__':
	if len(sys.argv) < 4:
		print "python app.py ckpt_top_dir host port"
		sys.exit(-1)

	ckpt_top_dir = sys.argv[1]
	#label_file is made up of lines containing ckpt_folder:ckpt_name
	with open(os.path.join(ckpt_top_dir,'ckpt_name.txt'),'r') as f:
		for line in f:
			_line = line.split(":")
			dic_ckpt_name[_line[0]] = _line[1]

	for ckpt in os.listdir(ckpt_top_dir):
		if ckpt == 'ckpt_name.txt':
			continue
		ckpt_dir = os.path.join(ckpt_top_dir, ckpt)
		label_file = os.path.join(ckpt_dir, 'labels.txt')

		print 'loading '+ckpt_dir+dic_ckpt_name[ckpt]+'...'
		pred = Predictor()
		pred.load_ckpt(ckpt_dir, dic_ckpt_name[ckpt], label_file)
		list_predictor.append(pred)
	print 'finished.'

	host = sys.argv[3]
	port = int(sys.argv[4])
	
	app.run(host=host, port=port, debug=False)

