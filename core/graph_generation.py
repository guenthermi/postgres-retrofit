#!/usr/bin/python3

import sys
import networkx as nx
import json
from collections import defaultdict
from itertools import combinations

import config
import db_connection as db_con


def get_schema(con, cur):
    rel_query = ("SELECT tc.table_name, tc.constraint_type, kcu.column_name, " +
                 "ccu.table_name AS foreign_table_name, " +
                 "ccu.column_name AS foreign_column_name " +
                 "FROM information_schema.table_constraints AS tc " +
                 "JOIN information_schema.key_column_usage AS kcu " +
                 "ON tc.constraint_name = kcu.constraint_name " +
                 "AND tc.table_schema = kcu.table_schema " +
                 "JOIN information_schema.constraint_column_usage AS ccu " +
                 "ON ccu.constraint_name = tc.constraint_name " +
                 "AND ccu.table_schema = tc.table_schema")
    cur.execute(rel_query)
    rels = cur.fetchall()
    schema = dict()
    for entry in rels:
        if not entry[0] in schema:
            schema[entry[0]] = {'pkey': None, 'fkeys': []}
        if entry[1] == 'PRIMARY KEY':
            schema[entry[0]]['pkey'] = entry[2]
        if entry[1] == 'FOREIGN KEY':
            f_rel = (entry[2], entry[3], entry[4])
            if 'fkeys' in schema[entry[0]]:
                schema[entry[0]]['fkeys'].append(f_rel)
    return schema

def construct_relation_graph(schema, columns, blacklist):
    result = nx.MultiDiGraph()
    bl = [[y.split('.') for y in x.split('~')] for x in blacklist]
    for key in columns.keys():
        result.add_node(key, columns=columns[key], pkey=schema[key]['pkey'])
    for table_name in schema.keys():
        fkeys = schema[table_name]['fkeys']
        if (len(fkeys) > 1) and (table_name not in columns):
            if not table_name in blacklist:
                relevant_fkeys = set()
                for key in fkeys:
                    if key[1] in columns.keys():
                        relevant_fkeys.add(key)
                for (key1, key2) in combinations(relevant_fkeys,2):
                    name = table_name
                    if len(fkeys) > 2:
                        name += ':' + key1[0] + '~' + key2[0]
                    result.add_edge(
                        key1[1],
                        key2[1],
                        col1=key1[0],
                        col2=key2[0],
                        name=table_name)
        if (len(fkeys) > 0) and (table_name in columns.keys()):
            for fkey in fkeys:
                if fkey[1] in columns.keys():
                    if not fkey[0] in blacklist:
                        result.add_edge(
                            table_name, fkey[1], col1=fkey[0], col2=fkey[2], name='-')
    return result


def get_all_db_columns(blacklists, con, cur):
    # string values
    query = ("SELECT table_name, column_name FROM information_schema.columns "
             + "WHERE table_schema = 'public' AND udt_name = 'varchar'")
    cur.execute(query)
    responses = cur.fetchall()
    names = defaultdict(list)
    for (table, col) in responses:
        if (not table in blacklists[0]) and (not (table + '.' + col) in blacklists[1]):
            names[table].append(col)
    return names


def main(argc, argv):

    # get database connection
    db_config = db_con.get_db_config(path=argv[2])
    con, cur = db_con.create_connection(db_config)

    # get retrofitting config
    conf = config.get_config(argv)

    # read out data from database
    db_columns = get_all_db_columns(
        (conf['TABLE_BLACKLIST'], conf['COLUMN_BLACKLIST']), con, cur)

    # construct graph from relational data
    schema = get_schema(con, cur)

    # export schema
    relation_graph = construct_relation_graph(
        schema, db_columns, conf['RELATION_BLACKLIST'])

    nx.write_gml(relation_graph, conf['SCHEMA_GRAPH_PATH'])

    return


if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
