import os
import sys
import random
import numpy as np
import json
from datetime import datetime
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, 'core/')
import db_connection as db_con

MODEL_SAVE_PATH = 'best_model.'

VEC_TABLE_TEMPL = '{vec_table}'
VEC_TABLE1_TEMPL = '{vec_table1}'
VEC_TABLE2_TEMPL = '{vec_table2}'

ITERATIONS = 10

OUTPUT_NAME = 'apps_cat_classify_'
OUTPUT_NAME_PLOT = 'apps_cat_classify_plot_'


def output_result(result, filename):
    f = open(filename, 'w')
    json.dump(result, f)
    return


def parse_bin_vec(data):
    vec = np.fromstring(bytes(data), dtype='float32')
    if np.linalg.norm(vec) < 0.001:
        vec += np.ones(vec.shape[0]) * 0.001  # prevent zero vecs
    return vec


def create_timestamp():
    return str(datetime.now().timestamp())


def get_vector_query(query_tmpl, table_names, vec_table_templ, vec_table1_templ, vec_table2_templ):
    query = ""
    if len(table_names) == 1:
        query = query_tmpl.replace(vec_table_templ, table_names[0])
    else:
        query = query_tmpl.replace(vec_table1_templ, table_names[0])
        query = query.replace(vec_table2_templ, table_names[1])
    return query


def classify(train_data, test_data, output_size):
    # get input data
    x = np.array([x[1] for x in train_data])
    y = np.array([x[0] for x in train_data])
    x_test = np.array([x[1] for x in test_data])
    y_test = np.array([x[0] for x in test_data])
    # create model
    backup_model_name = MODEL_SAVE_PATH + create_timestamp() + '.weights'
    model = Sequential()
    es = EarlyStopping(monitor='val_loss', mode='min', patience=50, verbose=1)
    model.add(Dense(600, input_dim=x.shape[1], activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(300, input_dim=600, activation='sigmoid'))
    model.add(Dense(output_size, activation='softmax'))
    checkpointer = ModelCheckpoint(
        filepath=backup_model_name, verbose=1, save_best_only=True, monitor='val_loss')
    # Compile model
    model.compile(loss='categorical_crossentropy',
                  optimizer='nadam', metrics=['accuracy'])
    model.fit(x, y, epochs=500, batch_size=8,
              validation_split=0.1, callbacks=[es, checkpointer])

    # Load best model and evaluate
    model.load_weights(backup_model_name)
    score = model.evaluate(x_test, y_test)
    res = model.predict(x_test)
    print('Score:', score[1])
    os.remove(backup_model_name)
    return score[1]


def construct_data_list(query_result):
    id_lookup = dict()
    value_lookup = dict()
    value_count = 0
    count = 0
    for elem in query_result:  # id value vec [vec2]
        if type(elem[1]) == type(None):
            continue
            count += 1
        if len(elem) == 3:  # one vectors
            value = elem[1]
            if value not in value_lookup:
                value_lookup[value] = value_count
                value_count += 1
            vec = parse_bin_vec(elem[2])
            if elem[0] in id_lookup:
                id_lookup[elem[0]][0].append(value_lookup[value])
            else:
                id_lookup[elem[0]] = (
                    [value_lookup[value]], vec / np.linalg.norm(vec))
        if len(elem) == 4:  # two vectors
            value = elem[1]
            if value not in value_lookup:
                value_lookup[value] = value_count
                value_count += 1
            v1 = parse_bin_vec(elem[2])
            v2 = parse_bin_vec(elem[3])
            combined = np.concatenate(
                [v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)])
            if elem[0] in id_lookup:
                id_lookup[elem[0]][0].append(value_lookup[value])
            else:
                id_lookup[elem[0]] = (
                    [value_lookup[value]], combined / np.linalg.norm(combined))
    # remove instances with multiple values
    id_lookup_clean = dict()
    for key in id_lookup:
        if len(set(id_lookup[key][0])) == 1:
            id_lookup_clean[key] = (
                list(set(id_lookup[key][0])), id_lookup[key][1])
    id_lookup = id_lookup_clean
    return id_lookup, value_lookup, value_count


def vec_dict2matrix(vecs, max_size=10000000):
    data = []
    key_data = list(vecs.keys())
    key_data = key_data[:max_size]
    for i in range(len(key_data)):
        data.append(vecs[key_data[i]][1])
    data = np.array(data)
    return key_data, data


def get_raw_data(query, con, cur):
    cur.execute(query)
    id_lookup, value_lookup, value_count = construct_data_list(cur.fetchall())
    id_keylist = list(id_lookup.keys())
    return id_lookup, value_lookup, value_count, id_keylist


def get_data(size, id_lookup, value_lookup, value_count, id_keylist):
    random.shuffle(id_keylist)
    train_data = dict()
    for elem in id_keylist[:size]:
        train_data[elem] = id_lookup[elem]

    key_data, data = vec_dict2matrix(train_data)
    for i, key in enumerate(key_data):
        value_encoded = np.zeros(value_count)
        for j in train_data[key][0]:
            value_encoded[j] = 1
        train_data[key] = (value_encoded, data[i])
    return train_data, value_lookup, value_count


def read_config_file(filename):
    f = open(filename, 'r')
    config = json.load(f)
    return config


def plot_graph(data, label, title, output_filename):
    labels = [label]
    fig1, ax1 = plt.subplots()
    ax1.set_title(title, fontsize=20)
    ax1.boxplot(data)
    plt.xticks(list(range(1, len(labels) + 1)), labels)
    plt.ylabel('Accuracy [%]', fontsize=18)
    plt.tick_params(labelsize=16)
    fig1.tight_layout()
    plt.savefig(output_filename)
    return


def main(argc, argv):
    db_config_file = argv[1]
    config = read_config_file(argv[2])

    train_data_size = int(config['train-size'])
    test_data_size = int(config['test-size'])
    output_folder = config['output_folder']

    db_config = db_con.get_db_config(db_config_file)
    con, cur = db_con.create_connection(db_config)

    # get query that returns tuple of relations which are present
    query = get_vector_query(
        config['query'], config['table_names'], VEC_TABLE_TEMPL, VEC_TABLE1_TEMPL, VEC_TABLE2_TEMPL)
    id_lookup, value_lookup, value_count, id_keylist = get_raw_data(
        query, con, cur)
    scores = []
    for i in range(ITERATIONS):
        data, value_lookup, output_size = get_data(
            train_data_size + test_data_size, id_lookup, value_lookup, value_count, id_keylist)
        input_data = list(data.values())
        random.shuffle(input_data)
        train_data = input_data[:train_data_size]
        test_data = input_data[train_data_size:]
        score = classify(train_data, test_data, output_size)
        scores.append(score)
        print('Scores:', scores)
        print('mu %f %% std %f %%' % (
            np.mean(scores), np.std(scores)))
    mu = np.mean(scores)
    std = np.std(scores)
    print('mu', mu, 'std', std)

    # output
    label = '+'.join(config['table_names'])
    # save output
    output_result(dict({label: {'mu': mu, 'std': std, 'scores': scores}}),
                  output_folder + OUTPUT_NAME + label + '.json')
    # plot output
    plot_graph(scores, label, 'App Category Classification',
               output_folder + OUTPUT_NAME_PLOT + label + '.png')
    return mu, std, scores


if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
