#!/usr/bin/python3

import sys

import config
from graph_generation import main as graph_generation
from group_extraction import main as group_extraction
from matrix_retrofit import main as retrofit
from vec2database import main as vec2database
from gml2json import main as gml2json

HELP_TEXT = (
    'USAGE: \033[1mrun_retrofitting.py\033[0m config_file db_config_file\n' +
    '\tconfig_file: json file that defines the retrofitting parameters' +
    '\tdb_config_file: json file that contains the database connection information')

def main(argc, argv):
    if argc < 3:
        print(HELP_TEXT)
        return

    conf = config.get_config(argv)
    print('Loaded config file', argv[1])
    retro_table_confs = conf['RETRO_TABLE_CONFS']
    print('Create schema graph in', conf['SCHEMA_GRAPH_PATH'], '...')
    graph_generation(argc, argv)
    gml2json(argc, argv) # only for visualization
    print('Extract groups and generate', conf['GROUPS_FILE_NAME'], '...')
    group_extraction(argc, argv)
    print('Start retrofitting ...')
    retrofit(argc, argv)
    for conf_path in retro_table_confs:
        print('Add retrofitted vectors to database as defined in', conf_path,
              '...')
        vec2database(3, [argv[0], conf_path, argv[2]])


if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
