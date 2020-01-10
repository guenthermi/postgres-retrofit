#!/usr/bin/python3

import sys
import psycopg2
import time
import numpy as np

from config import *
import db_connection as db_con

USE_BYTEA_TYPE = True


def init_tables(con, cur, table_name):
    # drop old table
    query_clear = "DROP TABLE IF EXISTS " + table_name + ";"
    result = cur.execute(query_clear)
    con.commit()
    print('Executed DROP TABLE on ' + table_name)

    # create table
    query_create_table = None
    if USE_BYTEA_TYPE:
        query_create_table = "CREATE TABLE " + table_name + \
            " (id serial PRIMARY KEY, word varchar, vector bytea);"
    else:
        query_create_table = "CREATE TABLE " + table_name + \
            " (id serial PRIMARY KEY, word varchar, vector float4[]);"
    result = cur.execute(cur.mogrify(query_create_table, (table_name, )))
    # commit changes
    con.commit()
    print('Created new table ' + table_name)

    return


def create_index(table_name, index_name, column_name, con, cur):
    query_drop = "DROP INDEX IF EXISTS " + index_name + ";"
    result = cur.execute(query_drop)
    con.commit()
    query_create_index = "CREATE INDEX " + index_name + \
        " ON " + table_name + " (" + column_name + ");"
    cur.execute(query_create_index)
    con.commit()
    print('Created index ' + str(index_name) + ' on table ' +
          str(table_name) + ' for column ' + str(column_name))


def create_bytea_array(array):
    vector = []
    for elem in array:
        try:
            vector.append(float(elem))
        except:
            return None
    return db_con.get_bytea(np.array(array, dtype='float32'))


def create_bytea_norm_array(array):
    vector = []
    for elem in array:
        try:
            vector.append(float(elem))
        except:
            return None
    return db_con.get_bytea(np.array(vector, dtype='float32') / np.linalg.norm(vector))


def insert_vectors(filename, con, cur, table_name, batch_size, normalized):
    f = open(filename, encoding="utf8")
    (_, size) = f.readline().split()
    d = int(size)
    count = 1
    line = f.readline()
    values = []
    while line:
        splits = line.split()
        vector = None
        if normalized:
            vector = create_bytea_norm_array(splits[1:])
        else:
            vector = create_bytea_array(splits[1:])
        if (len(splits[0]) > 0) and (vector !=
                                     None) and (len(splits) == (d + 1)):
            values.append({"word": splits[0], "vector": vector})
        else:
            print('parsing problem with ' + line)
            count -= 1
        if count % batch_size == 0:
            if USE_BYTEA_TYPE:
                cur.executemany(
                    "INSERT INTO " + table_name +
                    " (word,vector) VALUES (%(word)s, %(vector)s)",
                    tuple(values))
            else:
                cur.executemany("INSERT INTO " + table_name +
                                " (word,vector) VALUES (%(word)s, %(vector)s)",
                                tuple(values))
            con.commit()
            print('Inserted ' + str(count - 1) + ' vectors')
            values = []

        count += 1
        line = f.readline()

    if USE_BYTEA_TYPE:
        cur.executemany(
            "INSERT INTO " + table_name +
            " (word,vector) VALUES (%(word)s, %(vector)s)",
            tuple(values))
    else:
        cur.executemany("INSERT INTO " + table_name +
                        " (word,vector) VALUES (%(word)s, %(vector)s)",
                        tuple(values))
    con.commit()
    print('Inserted ' + str(count - 1) + ' vectors')
    values = []

    return


def main(argc, argv):
    # init db connection
    db_config = db_con.get_db_config(path=argv[2])
    con, cur = db_con.create_connection(db_config)

    if argc < 2:
        print('Configuration file for index creation required')
        return
    vec_config = json.load(open(argv[1], 'r'))

    init_tables(con, cur, vec_config['table_name'])

    insert_vectors(vec_config['vec_file_path'], con, cur,
                   vec_config['table_name'], db_config['batch_size'],
                   vec_config['normalized'])

    # commit changes
    con.commit()

    # create index
    create_index(vec_config['table_name'], vec_config['index_name'], 'word',
                 con, cur)

    # close connection
    con.close()


if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
