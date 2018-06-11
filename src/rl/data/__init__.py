from contextlib import contextmanager
from pkg_resources import resource_filename


def resource_path(resource_fname, resource_dir):
    resource_name = f'data/{resource_dir}/{resource_fname}'
    return resource_filename('rl', resource_name)


@contextmanager
def resource_open(resource_fname, resource_dir, local=True):

    if local:
        try:
            with open(resource_fname) as f:
                yield f
        except FileNotFoundError:
            pass
        else:
            return

    with open(resource_path(resource_fname, resource_dir)) as f:
        yield f
