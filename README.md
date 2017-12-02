# Tensorflow-Flask-Server
2017/12/2:
Add multi model support.
The ckpt_top_dir is like:
.
├── 1
│   ├── labels.txt
│   ├── model.ckpt-548706.data-00000-of-00001
│   ├── model.ckpt-548706.index
│   └── model.ckpt-548706.meta
├── 2
│   ├── labels.txt
│   ├── model.ckpt-269165.data-00000-of-00001
│   ├── model.ckpt-269165.index
│   └── model.ckpt-269165.meta
├── 3
│   ├── labels.txt
│   ├── model.ckpt-387137.data-00000-of-00001
│   ├── model.ckpt-387137.index
│   └── model.ckpt-387137.meta
└── ckpt_name.txt

python app.py ckpt_top_dir host port

when in use:
http://host:port/flask/api/v1.0/tasks?img_url=……
