#!flask/bin/python
import urllib
import sys
import io
from PIL import Image

from predict import Predictor
from flask import Flask, jsonify, request

num_classes = 100
predictor = Predictor()
type_list = []

app = Flask(__name__)

#init type_list
def get_typelist(label_file):
	_list = []
	with open(label_file,'r') as f:
		for line in f:
			_type = line.split(":")
			if len(_type)>1:
				_list.append(_type[1].strip())
	return _list
	
@app.route('/flask/api/v1.0/tasks', methods=['GET'])
def get_tasks():
	try:
		url = urllib.unquote(request.args.get('img_url'))
		data = urllib.urlopen(url).read()
		img = Image.open(io.BytesIO(data))
		
		pred = predictor.get_predict(img)
		res_dic = {type_list[i]:float(pred[i]) for i in range(num_classes)}
		return jsonify(res_dic)		
	except Exception,e:
		print e
		return '{}'

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'})

if __name__ == '__main__':
	if len(sys.argv) < 5:
		print "python app.py ckpt_file label_file host port"
		sys.exit(-1)

	predictor.load_ckpt(sys.argv[1])
	print "load finished."
	type_list = get_typelist(sys.argv[2])
	host = sys.argv[3]
	port = int(sys.argv[4])
	
	app.run(host=host, port=port, debug=False)

