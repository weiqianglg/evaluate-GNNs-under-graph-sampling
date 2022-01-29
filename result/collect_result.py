from collections import defaultdict

import pandas as pd

from itertools import product
import sqlite3
from result.result_file import get_result_files, result_title


def node_classification_old_result2db(df_path, db_name):
    conn = sqlite3.connect(f"{db_name}.db")
    c = conn.cursor()
    try:
        c.execute(f"DROP TABLE {db_name}")
    except sqlite3.OperationalError as e:
        print(e)

    c.execute(
        f"CREATE TABLE {db_name} (id INTEGER,dataset TEXT, model TEXT, sample TEXT, induced TEXT, p REAL, metric REAL)")

    df = pd.read_excel(df_path, sheet_name="detail")
    records = []
    total_index = 0

    for _, row in df.iterrows():
        total_index += 1
        induced = row['induced']
        records.append((
            total_index,
            row['dataset'].lower(),
            row['model'],
            row['sample'],
            induced,
            row['sampled_ratio'],
            row['test_acc_mean']
        ))
    print(df_path, total_index)
    c.executemany(f"INSERT INTO {db_name} VALUES (?,?,?,?,?,?,?)", records)

    conn.commit()
    conn.close()


def files_update_db(task_folder, db_name):
    conn = sqlite3.connect(f"{db_name}.db")
    c = conn.cursor()
    for df_path in get_result_files(task_folder):
        _, __, sample_method, induced = result_title(df_path, True)
        induced = "True" in induced
        new_results = defaultdict(list)
        df = pd.read_excel(df_path)
        for _, row in df.iterrows():
            key = (row['dataset'].lower(),
                   row['model'],
                   sample_method,
                   induced,
                   row['left_p'])
            new_results[key].append(row['test_acc'])
        total_index = 0
        records = []
        for key, metrics in new_results.items():
            c.execute(
                f"DELETE FROM {db_name} WHERE dataset=? AND model=? AND sample=? AND induced=? AND p=?",
                key)
            for acc in metrics:
                total_index += 1
                records.append((
                    total_index,
                    key[0],
                    key[1],
                    key[2],
                    key[3],
                    key[4],
                    acc
                ))
        c.executemany(f"INSERT INTO {db_name} VALUES (?,?,?,?,?,?,?)", records)
        conn.commit()
    conn.close()


def files2db(task_folder, db_name):
    conn = sqlite3.connect(f"{db_name}.db")
    c = conn.cursor()
    try:
        c.execute(f"DROP TABLE {db_name}")
    except sqlite3.OperationalError as e:
        print(e)

    c.execute(
        f"CREATE TABLE {db_name} (id INTEGER,dataset TEXT, model TEXT, sample TEXT, induced TEXT, p REAL, metric REAL)")

    total_index = 0
    for df_path in get_result_files(task_folder):

        records = []
        _, __, sample_method, induced = result_title(df_path, True)
        induced = "True" in induced

        df = pd.read_excel(df_path)
        cnt = 0
        for _, row in df.iterrows():
            total_index += 1
            cnt += 1
            records.append((
                total_index,
                row['dataset'].lower(),
                row['model'],
                sample_method,
                induced,
                row['left_p'],
                row['test_acc']
            ))
        print(df_path, cnt)
        c.executemany(f"INSERT INTO {db_name} VALUES (?,?,?,?,?,?,?)", records)

        conn.commit()
    conn.close()


def find_missing(db_name, datasets, model=None):
    conn = sqlite3.connect(f"{db_name}.db")
    c = conn.cursor()

    model = ['GAT', "GatedGCN", "GCN", "GIN", "GraphSage", "MLP", "MoNet"] if model is None else model
    sample = ['BreadthFirstSearchSampler', 'RandomWalkSampler', 'ForestFireSampler',
              'MetropolisHastingsRandomWalkSampler']
    induced = ['0', '1']
    percent = [0.1, 0.2, 0.3, 0.4, 0.5, 1]
    for d, m, s, i, p in product(datasets, model, sample, induced, percent):
        c.execute(f"SELECT COUNT(*) FROM {db_name} WHERE dataset=? AND model=? AND sample=? AND induced=? AND p=?",
                  (d, m, s, i, p))
        r = c.fetchone()
        if r[0] < 4:
            print('lack', d, m, s, i, p, r[0])
        elif r[0] > 4:
            continue
            print('more', d, m, s, i, p, r[0])
            c.execute(f"SELECT metric FROM {db_name} WHERE dataset=? AND model=? AND sample=? AND induced=? AND p=?",
                      (d, m, s, i, p))

            print('  ', sorted([m[0] for m in c.fetchall()]))


if __name__ == '__main__':
    task_folder = "../out/node_classification_0813"
    task_type = "gated_simple_node_classification"
    # node_classification_old_result2db("simple_node_classification_all_mean_std.xlsx", "old_nc")
    files_update_db(task_folder, task_type)
    # files2db(task_folder, task_type)

    # ds = ['cifar10', "mnist"]
    # ds = ['ogbn-arxiv']
    # ds = ['actor', 'citeseer', 'cora', 'pubmed']
    # ds = ["ogbl-collab"]
    ds = ['cluster']
    # find_missing(task_type, ds)
