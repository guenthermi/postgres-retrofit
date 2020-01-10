import json

DEFAULT_FILE_NAME = '../config/retro_config.json'


def get_config_by_file_name(file_name):
    f = open(file_name, 'r')
    configuration = json.load(f)
    f.close()
    return configuration


def get_config(argv=[]):
    file_name = DEFAULT_FILE_NAME
    if len(argv) > 1:
        file_name = argv[1]
    return get_config_by_file_name(file_name)
