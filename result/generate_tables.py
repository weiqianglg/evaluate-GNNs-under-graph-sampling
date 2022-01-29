from itertools import product
import sqlite3
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import seaborn as sns
import matplotlib.pyplot as plt


def get_mean_std_string(row):
    mean, std = row['metric_mean'], row['metric_std']
    mean, std = mean * 100, std * 100
    return f"{mean:.1f}±{std:.1f}"


def mean_std(results):
    acc = results.groupby(['dataset', 'model', 'sample', 'induced', 'p'])['metric']
    means, stds = acc.mean(), acc.std()
    _performance = pd.concat([means, stds], axis=1)
    _performance.columns = ['metric_mean', 'metric_std']

    ind = _performance.index.to_frame(index=False)
    value = _performance.reset_index(drop=True)
    detail_performance = pd.concat([ind, value], axis=1, ignore_index=True)
    detail_performance.columns = ['dataset', 'model', 'sample', 'induced', 'p', 'metric_mean', 'metric_std']

    return detail_performance


def db2df(db_name):
    engine = create_engine(f'sqlite:///{db_name}.db')
    df = pd.read_sql_table(f"{db_name}", engine)
    return df


def calc_mean_std(db_name):
    df = db2df(db_name)
    y = df[df['induced'] == '0']
    y = y[y['p'] == .5]
    # y = y[y['metric'] != 0.5]
    y = mean_std(y)
    y['result'] = y.apply(get_mean_std_string, axis=1)
    y = y.pivot(index=['dataset', 'sample'], columns='model', values='result')
    y.to_excel(f'table3r0.5_{db_name}.xlsx')
    return y


def all_mean_std():
    db_names = ["gated_simple_node_classification", "arxiv_node", "cluster_node", "simple_link_prediction",
                "collab_link_prediction", "graph_classification"]
    metric = pd.DataFrame()
    for db_name in db_names:
        df = db2df(db_name)
        y = df[df['induced'] == '0']
        y = y[y['p'] == .3]
        y = mean_std(y)
        metric = metric.append(y, ignore_index=True)
    metric.to_excel(f'all_mean_std_for_r0.3_part.xlsx')
    return metric


def induced_improve(db_name='', dataset='Actor', model='GraphSage', p=0.3, dataset_name=''):
    # db_names = ["gated_simple_node_classification", "arxiv_node", "cluster_node", "simple_link_prediction",
    #             "collab_link_prediction", "graph_classification"]
    # db_name = 'arxiv_node'
    y = db2df(db_name)
    y = y[y['p'] == p]
    y = y[y['dataset'] == dataset]
    y = y[y['model'] == model]
    y = mean_std(y)
    improve = pd.DataFrame()
    g = y.groupby(['sample', 'p'])
    for (s_, r_), index in g.groups.items():
        acc = y.loc[index]
        delta = acc[acc['induced'] == '1']['metric_mean'].item() / acc[acc['induced'] == '0'][
            'metric_mean'].item() - 1
        improve = improve.append({'sample': s_, 'sampled_ratio': r_, 'delta': delta}, ignore_index=True)
    improve['delta'] = 100 * improve['delta']
    improve['dataset'] = dataset_name if dataset_name else db_name + dataset
    return improve


def all_induced_improve(ratio=0.1):
    result = pd.DataFrame()

    imp = induced_improve("gated_simple_node_classification", "actor", "GatedGCN", p=ratio, dataset_name="Actor (NC)")
    result = result.append(imp, ignore_index=True)
    imp = induced_improve("old_nc", "cora", "GCN", p=ratio, dataset_name="Cora (NC)")
    result = result.append(imp, ignore_index=True)
    imp = induced_improve("old_nc", "citeseer", "GCN", p=ratio, dataset_name="CiteSeer (NC)")
    result = result.append(imp, ignore_index=True)
    imp = induced_improve("old_nc", "pubmed", "GCN", p=ratio, dataset_name="Pubmed (NC)")
    result = result.append(imp, ignore_index=True)

    imp = induced_improve("arxiv_node", "ogbn-arxiv", "GIN", p=ratio, dataset_name="ARXIV")
    result = result.append(imp, ignore_index=True)
    imp = induced_improve("cluster_node", "cluster", "MLP", p=ratio, dataset_name="CLUSTER")
    result = result.append(imp, ignore_index=True)

    imp = induced_improve("simple_link_prediction", "actor", "GatedGCN", p=ratio, dataset_name="Actor (LP)")
    result = result.append(imp, ignore_index=True)
    imp = induced_improve("simple_link_prediction", "cora", "GraphSage", p=ratio, dataset_name="Cora (LP)")
    result = result.append(imp, ignore_index=True)
    imp = induced_improve("simple_link_prediction", "citeseer", "GraphSage", p=ratio, dataset_name="CiteSeer (LP)")
    result = result.append(imp, ignore_index=True)
    imp = induced_improve("simple_link_prediction", "pubmed", "MoNet", p=ratio, dataset_name="Pubmed (LP)")
    result = result.append(imp, ignore_index=True)
    imp = induced_improve("collab_link_prediction", "ogbl-collab", "GraphSage", p=ratio, dataset_name="COLLAB")
    result = result.append(imp, ignore_index=True)
    imp = induced_improve("graph_classification", "mnist", "MoNet", p=ratio, dataset_name="MNIST")
    result = result.append(imp, ignore_index=True)
    imp = induced_improve("graph_classification", "cifar10", "GCN", p=ratio, dataset_name="CIFAR10")
    result = result.append(imp, ignore_index=True)
    result.to_excel(f"induced{ratio}.xlsx")
    return result


def xlxs2table5(mode='sample'):
    y = pd.read_excel('all_mean_std_for_r0.3.xlsx', index_col=0, sheet_name='small')
    y = y[['dataset', 'sample', 'p', 'model', 'metric_mean']]
    y['rank'] = -1
    group, measure = ('model', 'sample') if mode == 'sample' else ('sample', 'model')
    g = y.groupby(['dataset', group, 'p'])
    for index in g.groups.values():
        acc = y.loc[index]
        y.loc[index, 'metric_mean'] /= y['metric_mean'][index].max()
        sorted_acc = acc.sort_values('metric_mean', ascending=False)
        y.loc[sorted_acc.index, 'rank'] = range(1, sorted_acc.shape[0] + 1)
    d = y.groupby([measure])[['metric_mean', 'rank']].mean()
    d.to_excel('./table5.xlsx')
    return d


def plot_induced_improve(xls_path='induced0.1.xlsx'):
    y = pd.read_excel(xls_path, index_col=0, sheet_name='matrix')
    print(y)
    ax = sns.heatmap(y, annot=True, cmap='YlGnBu')
    plt.tight_layout()
    fig = ax.get_figure()
    fig.savefig(f'{xls_path}.png', dpi=800)
    plt.show()


if __name__ == '__main__':
    task_type = "gated_simple_node_classification"
    # task_type = "arxiv_node"
    # task_type = "cluster_node"
    # task_type = "simple_link_prediction"
    # task_type = "graph_classification"
    task_type = "collab_link_prediction"
    # y = calc_mean_std(task_type)
    # all_mean_std()
    # xlxs2table5()
    # result = all_induced_improve()
    # plot_induced_improve()
