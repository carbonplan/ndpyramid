import importlib


def get_version():
    try:
        return importlib.import_module('ndpyramid').__version__
    except ModuleNotFoundError:
        return '-9999'


def multiscales_template(datasets=[], type='', method='', version='', args=[], kwargs={}):
    # https://forum.image.sc/t/multiscale-arrays-v0-1/37930
    d = [
        {
            "datasets": datasets,
            "type": type,
            "metadata": {"method": method, "version": version, "args": args, "kwargs": kwargs},
        }
    ]
    return d
