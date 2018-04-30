from contextlib import contextmanager
from pkg_resources import resource_filename


@contextmanager
def open_resource(resource_fname, resource_dir, local=True):
    if local:
        try:
            with open(resource_fname) as f:
                yield f
        except FileNotFoundError:
            pass

    resource_name = f'data/{resource_dir}/{resource_fname}'
    with open(resource_filename('rl', resource_name)) as f:
        yield f
