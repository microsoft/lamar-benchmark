import json

def same_configs(config, path):
    if not path.exists():
        return False
    config_existing = read_config(path)
    return config == config_existing


def read_config(path):
    with open(path, 'r') as fid:
        return json.load(fid)


def write_config(config, path):
    with open(path, 'w') as fid:
        json.dump(config, fid, indent=4)
