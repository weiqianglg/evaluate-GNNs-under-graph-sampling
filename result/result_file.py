from os import path as osp
from pathlib import Path


def get_result_files(result_dir="./out/node_classification", pattern="*.xlsx"):
    p = Path(result_dir)
    return [str(path) for path in p.glob(pattern)]


def result_title(result_path, return_verbose=False):
    title = osp.basename(result_path)[:-len('.xlsx')]
    if return_verbose:
        s = title.split('-')
        ds, method, sample, induced = s[:-3], s[-3], s[-2], s[-1]
        return "".join(ds), method, sample, induced
    else:
        return title

