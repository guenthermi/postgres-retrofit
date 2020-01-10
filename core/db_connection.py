import json
import os
import psycopg2

DB_CONFIG_PATH = os.path.dirname(__file__) + '/../config/db_config.json'


def get_db_config(path=DB_CONFIG_PATH):
    f_db_config = open(path, 'r')
    db_config = json.loads(f_db_config.read())
    f_db_config.close()
    return db_config

def get_bytea(value):
    return psycopg2.Binary(value)

def create_connection(db_config):
    con = None
    cur = None
    # create db connection
    try:
        con = psycopg2.connect(
            "dbname='" + db_config['db_name'] + "' user='" +
            db_config['username'] + "' host='" + db_config['host'] +
            "' password='" + db_config['password'] + "'")
    except:
        print('ERROR: Can not connect to database')
        print("dbname='" + db_config['db_name'] + "' user='" +
              db_config['username'] + "' host='" + db_config['host'] +
              "' password='" + db_config['password'] + "'")
        return
    cur = con.cursor()
    return con, cur
