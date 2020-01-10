import os
import re
import json
import base64
import numpy as np
from collections import defaultdict

import db_connection as db_con


def parse_groups(group_filename, vectors_encoded=True):
    f = open(group_filename)
    groups = json.load(f)
    for key in groups:
        for group in groups[key]:
            for key in group['elements']:
                if group['type'] == 'categorial':
                    if vectors_encoded:
                        group['elements'][key]['vector'] = np.fromstring(base64.decodestring(
                            bytes(group['elements'][key]['vector'], 'ascii')), dtype='float32')
                    else:
                        group['elements'][key]['vector'] = group['elements'][key]['vector']
            if 'inferred_elements' in group:
                if group['type'] == 'categorial':
                    for key in group['inferred_elements']:
                        if vectors_encoded:
                            group['inferred_elements'][key]['vector'] = np.fromstring(base64.decodestring(
                                bytes(group['inferred_elements'][key]['vector'], 'ascii')), dtype='float32')
                        else:
                            group['inferred_elements'][key]['vector'] = group['inferred_elements'][key]['vector']
    return groups


def get_data_columns_from_group_data(groups):
    result = set()
    for key in groups:
        if groups[key][0]['type'] == 'categorial':
            result.add(key)
    return list(result)


def get_column_data_from_label(label, type):
    if type == 'column':
        try:
            table_name, column_name = label.split('.')
            return table_name, column_name
        except:
            print('ERROR: Can not decode %s into table name and column name' %
                  (label))
            return
    if type == 'relation':
        try:
            c1, c2 = label.split('~')
            c1_table_name, c1_column_name = c1.split('.')
            c2_table_name, c2_column_name = c2.split('.')
            return c1_table_name, c1_column_name, c2_table_name, c2_column_name
        except:
            print('ERROR: Can not decode relation label %s ' % (label))
            return


def get_label(x, y): return '%s#%s' % (x, y)


def tokenize_sql_variable(name):
    return "regexp_replace(%s, '[\.#~\s\xa0,\(\)/\[\]:]+', '_', 'g')" % (name)


def tokenize(term):
    if type(term) == str:
        return re.sub('[\.#~\s,\(\)/\[\]:]+', '_', str(term))
    else:
        return ''


def get_terms(columns, con, cur):
    result = dict()
    for column in columns:
        table_name, column_name = column.split(
            '.')  # TODO get this in an encoding save way
        # construct sql query
        sql_query = "SELECT %s FROM %s" % (column_name, table_name)
        cur.execute(sql_query)
        result[column] = [tokenize(x[0]) for x in cur.fetchall()]
        result[column] = list(set(result[column]))  # remove duplicates
    return result


def construct_index_lookup(list_obj):
    result = dict()
    for i in range(len(list_obj)):
        result[list_obj[i]] = i
    return result


def get_dist_params(vectors):
    # returns the distribution parameter for vector elments
    m_value = 0
    count = 0
    values = []
    for key in vectors:
        max_inst = 0
        for term in vectors[key]:
            m_value += np.mean(vectors[key][term])
            values.extend([x for x in vectors[key][term]])
            max_inst += 1
            count += 1
            if max_inst > 100:
                break
    m_value /= count
    s_value = np.mean((np.array(values) - m_value)**2)
    return m_value, s_value


def execute_threads_from_pool(thread_pool, verbose=False):
    while(len(thread_pool) > 0):
        try:
            next = thread_pool.pop()
            if verbose:
                print('Number of threads:', len(thread_pool))
            next.start()
            next.join()
        except:
            print("Warning: threadpool.pop() failed")
    return


def get_vectors_for_present_terms_from_group_file(data_columns, groups_info):
    result_present = dict()
    dim = 0
    for column in data_columns:
        group = groups_info[column][0]['elements']
        group_extended = groups_info[column][0]['inferred_elements']
        result_present[column] = dict()
        for term in group:
            result_present[column][term] = np.array(
                group[term]['vector'], dtype='float32')
            dim = len(result_present[column][term])
        for term in group_extended:
            result_present[column][term] = np.array(
                group_extended[term]['vector'], dtype='float32')
            dim = len(result_present[column][term])
    return result_present, dim


def get_terms_from_vector_set(vec_table_name, con, cur):
    QUERY_TMPL = "SELECT word, vector, id FROM %s WHERE id >= %d AND id < %d"
    BATCH_SIZE = 500000
    term_dict = dict()
    min_id = 0
    max_id = BATCH_SIZE
    while True:
        query = QUERY_TMPL % (vec_table_name, min_id, max_id)
        cur.execute(query)
        term_list = [x for x in cur.fetchall()]
        if len(term_list) < 1:
            break
        for (term, vector, freq) in term_list:
            splits = term.split('_')
            current = [term_dict, None, -1]
            i = 1
            while i <= len(splits):
                subterm = '_'.join(splits[:i])
                if subterm in current[0]:
                    current = current[0][subterm]
                else:
                    current[0][subterm] = [dict(), None, -1]
                    current = current[0][subterm]
                i += 1
            current[1] = vector
            current[2] = freq
        min_id = max_id
        max_id += BATCH_SIZE
    return term_dict
