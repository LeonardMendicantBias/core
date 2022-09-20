import os
import h5py
from flask import Flask, jsonify

from models import Task, Split

app = Flask(__name__)


@app.route('/tasks', methods=['GET'])
def get_tasks():
    # tasks = os.listdir("/mount/dataset/")
    tasks = os.listdir("/home/leonard/deep_learning/core/data/result/")
    json = jsonify({'tasks': [t.split('.')[0] for t in tasks]})
    return json 

@app.route('/tasks/<name>', methods=['GET'])
def task(name):
    d = Task.to_arda(name)
    return jsonify(d)

@app.route('/tasks/<name>/<split>', methods=['GET'])
def get_split(name, split):
    d = Split.to_arda_2(name, split)
    return jsonify(d)

# @app.route('/tasks/<name>/<split>/<sname>', methods=['GET'])
# def get_split(name, split, sname):
#     d = Task.Split.to_arda(name, split, sname)
#     return jsonify(d)

# @app.route('/datasets/<dataset_name>', methods=['GET'])
# def get_dataset(dataset_name=0):
#     with h5py.File(f'/mount/dataset/{dataset_name}', "r") as fp:
#         for name in fp: 
#             print(name)
#     return jsonify({'dataset': dataset_name})

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=8000,
        debug=True
    )
# @app.route('/datasets/<dataset_name>', methods=['GET'])
# dfe get_