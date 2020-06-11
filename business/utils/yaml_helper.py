import yaml


def get_data_from_yaml(yaml_path):
    with open(yaml_path, encoding='utf-8') as normal_file:
        res = yaml.load(normal_file, Loader=yaml.FullLoader)
    return res
