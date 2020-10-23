
import json
import networkx as nx
import numpy as np
from collections import defaultdict
from node2vec import Node2Vec

import retro_utils as utils
import config
import db_connection as db_con

def get_db_vectors(table_name, con, cur, freq_limit=0, return_frequencies=False):
    result = dict()
    filtered = dict()
    query = "SELECT id, word, vector FROM %s" % (table_name,)
    cur.execute(query)

    for (id, term, vec_bytes) in cur.fetchall():
        if int(id) > freq_limit:
            if (return_frequencies):
                result[term] = (np.fromstring(bytes(vec_bytes), dtype='float32'), id)
            else:
                result[term] = np.fromstring(bytes(vec_bytes), dtype='float32')
        else:
            if (return_frequencies):
                filtered[term] = (np.fromstring(bytes(vec_bytes), dtype='float32'), id)
            else:
                filtered[term] = np.fromstring(bytes(vec_bytes), dtype='float32')
    if freq_limit != 0:
        return result, filtered
    else:
        return result

def get_vector_query(query_tmpl, table_names, vec_table_templ, vec_table1_templ, vec_table2_templ):
    query = ""
    if len(table_names) == 1:
        query = query_tmpl.replace(vec_table_templ, table_names[0])
    else:
        query = query_tmpl.replace(vec_table1_templ, table_names[0])
        query = query.replace(vec_table2_templ, table_names[1])
    return query

def export_edgelist(term_list, edges, file_name_edgelist, file_name_termlist):
    G = nx.Graph()
    lookup = utils.construct_index_lookup(term_list)
    for (a, b) in edges:
        G.add_edge(lookup[a], lookup[b])
    nx.edgelist.write_edgelist(G, file_name_edgelist, data=False)
    f = open(file_name_termlist, 'w')
    f.writelines(['%d %s\n' % (i, term) for i, term in enumerate(term_list)])
    f.close()
    return


def get_data_columns(groups_info):
    data_columns = []
    for key in groups_info:
        for group in groups_info[key]:
            if group['type'] == 'categorial':
                data_columns.append(group['name'])
    return data_columns


def get_term_list(all_terms):
    term_list = []
    for key in all_terms:
        for term in all_terms[key]:
            row_label = utils.get_label(key, term)
            term_list.append(row_label)
    return term_list


def get_edges(term_list, groups_info, con, cur):
    edges = defaultdict(int)
    for key in groups_info:
        for group in groups_info[key]:
            if group['type'] == 'relational':
                t1, c1, t2, c2 = utils.get_column_data_from_label(
                    key, 'relation')
                cur.execute(group['query'])
                column_name1 = '%s.%s' % (t1, c1)
                column_name2 = '%s.%s' % (t2, c2)
                c = 0
                for (a, b) in cur.fetchall():
                    term1 = utils.get_label(
                        column_name1, utils.tokenize(a))
                    term2 = utils.get_label(
                        column_name2, utils.tokenize(b))
                    edges[(term1, term2)] += 1
            if group['type'] == 'categorial':
                t1, c1, = utils.get_column_data_from_label(key, 'column')
                cur.execute("SELECT %s FROM %s" % (c1, t1))
                prefix = '%s.%s' % (t1, c1)
                for (a,) in cur.fetchall():
                    term = utils.get_label(prefix, utils.tokenize(a))
                    edges[(term, prefix)] += 1
    return edges

def create_edgelist(retro_config_filename, db_config_filename, graph_name):
    conf = config.get_config_by_file_name(retro_config_filename)

    groups_info = utils.parse_groups(conf['GROUPS_FILE_NAME'])

    db_config = db_con.get_db_config(path=db_config_filename)
    con, cur = db_con.create_connection(db_config)

    data_columns = get_data_columns(groups_info)
    all_terms = utils.get_terms(data_columns, con, cur)
    term_list = get_term_list(all_terms)
    term_list += data_columns
    edges = get_edges(term_list, groups_info, con, cur)

    export_edgelist(term_list, edges, graph_name + '.edges', graph_name + '.terms')
    return

