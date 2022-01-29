import os.path as osp
import seaborn
import matplotlib.pyplot as plt
import pandas as pd
from result.result_file import result_title, get_result_files


def _keep_precise(x):
    return round(x, 2)


def _load_result(result_path, no_zero_filter='test_acc', sampled_ratio=None):
    print("load", result_path)
    result = pd.read_excel(result_path)
    result = result[result[no_zero_filter] > 0]
    if sampled_ratio is not None:
        result['sampled_ratio'] = result[sampled_ratio].apply(_keep_precise)
    else:
        ratio = result['sampled_V'] / result.iloc[-1]['sampled_V']
        ratio = ratio.apply(_keep_precise)
        result['sampled_ratio'] = ratio
    return result


def _merge_result(result_paths, return_verbose=False, no_zero_filter='test_acc', sampled_ratio=None):
    results = []
    for i, result_path in enumerate(result_paths):
        result = _load_result(result_path, no_zero_filter, sampled_ratio=sampled_ratio)
        if return_verbose:
            dataset, model, sample, induced = result_title(result_path, return_verbose=True)
            result['dataset'] = dataset
            result['sample'] = sample
            result['induced'] = induced
        else:
            model = result_title(result_path)
        result['model'] = model
        results.append(result)
    results = pd.concat(results)
    return results


def plot_label_entropy_box(result_paths, *, save_dir=None, sampled_ratio=None):
    result_paths = [result_paths] if isinstance(result_paths, str) else result_paths
    plt.close()
    results = _merge_result(result_paths, no_zero_filter='te', return_verbose=True)
    results = results[results['sampled_ratio'].isin([0.1, 0.2, 0.3, 0.4, 0.5])]
    results = results.rename(columns={'sample': 'sampling method', 'te': 'KL divergence'})
    ax = seaborn.boxplot(x='sampled_ratio', y='KL divergence', data=results, hue='sampling method', showmeans=False)
    # ax.axhline(1.92589, ls='dashed', color='k') # unsampled entropy
    # plt.text(0.05, 1.93, 'entropy of unsampled training labels')
    ax.set(ylim=(0, 1))
    plt.xlabel("$r$")
    plt.tight_layout()
    if save_dir:
        fig = ax.get_figure()
        fig.savefig(osp.join(save_dir, f'pubmed_label_entropy.png'), dpi=800)
    else:
        plt.show()


def plot_box(result_paths, *, save_dir=None, auto_ylim=True, sampled_ratio=None):
    result_paths = [result_paths] if isinstance(result_paths, str) else result_paths
    plt.close()
    results = _merge_result(result_paths, sampled_ratio=sampled_ratio, return_verbose=True)
    title = ' vs '.join(map(result_title, result_paths))

    ax = seaborn.boxplot(x='sampled_ratio', y='test_acc', data=results, hue='model', showmeans=True)
    if not auto_ylim:
        ax.set(ylim=(0, 1))  # make all graph have the same y scope
    # ax.set(title=title)
    plt.xlabel("$r$")
    plt.ylabel("F1")
    if save_dir:
        fig = ax.get_figure()
        fig.savefig(osp.join(save_dir, f'{title}.png'), dpi=400)
    else:
        plt.show()


def find_best2_and_worst2_gnn_by_mean(result_paths, *, save_dir=None):
    results = _merge_result(result_paths)
    ratios = results['sampled_ratio'].unique()
    acc = results.groupby(['model', 'sampled_ratio'])['test_acc']
    means = acc.mean()

    gnn_performance = pd.DataFrame()
    for ratio in ratios:
        y = means.xs(ratio, level=1).sort_values(ascending=False)
        gnn_performance = gnn_performance.append(
            pd.DataFrame([[ratio, y.index[0], y[0], 1],
                          [ratio, y.index[1], y[1], 2],
                          [ratio, y.index[-2], y[-2], -2],
                          [ratio, y.index[-1], y[-1], -1]],
                         columns=['sampled_ratio', 'model', 'test_acc_mean', 'rank']),
            ignore_index=True
        )
    return gnn_performance


def write_mean_std(result_paths, *, save_dir=None, detail=True, sampled_ratio=None):
    results = _merge_result(result_paths, return_verbose=True, sampled_ratio='left_p')
    acc = results.groupby(['dataset', 'model', 'sample', 'induced', 'sampled_ratio'])['test_acc']
    means, stds = acc.mean(), acc.std()
    _performance = pd.concat([means, stds], axis=1)
    _performance.columns = ['test_acc_mean', 'test_acc_std']
    if save_dir:
        writer = pd.ExcelWriter(osp.join(save_dir, "all_mean_std.xlsx"))
        _performance.to_excel(writer, sheet_name="simple")
        if detail:
            ind = _performance.index.to_frame(index=False)
            value = _performance.reset_index(drop=True)
            detail_performance = pd.concat([ind, value], axis=1, ignore_index=True)
            detail_performance.to_excel(writer, sheet_name="detail")
        writer.save()
        writer.close()
    return _performance


def folder_single_plot():
    rps = get_result_files()
    for rp in rps:
        print(rp)
        plot_box(rp, save_dir="./fig")


def get_mean_std_string(row):
    mean, std = row['test_acc_mean'], row['test_acc_std']
    mean, std = mean * 100, std * 100
    return f"{mean:.1f}±{std:.1f}"


def xlxs2table2():
    t = pd.read_excel('./out/all_mean_std.xlsx', index_col=0, sheet_name='detail')
    y = t[t['induced'] == False]
    y = y[y['sample'] == 'MetropolisHastingsRandomWalkSampler']
    y = y[y['model'] == 'GCN']
    y['f1'] = y.apply(get_mean_std_string, axis=1)
    y = y[['dataset', 'sampled_ratio', 'f1']]
    y.to_excel('./out/table2.xlsx')
    return y


def xlxs2table3():
    t = pd.read_excel('simple_node_classification_all_mean_std.xlsx', index_col=0, sheet_name='detail')
    y = t[t['induced'] == False]
    y = y[y['sampled_ratio'] == 0.3]
    y['f1'] = y.apply(get_mean_std_string, axis=1)
    y = y[['dataset', 'model', 'sample', 'f1']]
    y = y.pivot(index=['dataset', 'sample'], columns='model', values='f1')
    y.to_excel('table3r0.3.xlsx')
    return y


def xlxs2table4():
    t = pd.read_excel('./out/all_mean_std.xlsx', index_col=0, sheet_name='detail')
    y = t[t['induced'] == False]
    y = y[y['sampled_ratio'] >= 0.35]
    y = y[y['sampled_ratio'] < 1.0]
    y = y[['dataset', 'sample', 'sampled_ratio', 'model', 'test_acc_mean']]
    y['rank'] = -1
    g = y.groupby(['dataset', 'sample', 'sampled_ratio'])
    for index in g.groups.values():
        acc = y.loc[index]
        y.loc[index, 'test_acc_mean'] /= y['test_acc_mean'][index].max()
        sorted_acc = acc.sort_values('test_acc_mean', ascending=False)
        y.loc[sorted_acc.index, 'rank'] = range(1, sorted_acc.shape[0] + 1)
    d = y.groupby(['model'])[['test_acc_mean', 'rank']].mean()
    d.to_excel('./out/table4.xlsx')
    return d


def xlxs2table5():
    y = pd.read_excel('./all_mean_std_for_r0.3.xlsx', index_col=0)
    y = y[['dataset', 'sample', 'sampled_ratio', 'model', 'test_acc_mean']]
    y['rank'] = -1
    g = y.groupby(['dataset', 'model', 'sampled_ratio'])
    for index in g.groups.values():
        acc = y.loc[index]
        y.loc[index, 'test_acc_mean'] /= y['test_acc_mean'][index].max()
        sorted_acc = acc.sort_values('test_acc_mean', ascending=False)
        y.loc[sorted_acc.index, 'rank'] = range(1, sorted_acc.shape[0] + 1)
    d = y.groupby(['sample'])[['test_acc_mean', 'rank']].mean()
    d.to_excel('./table5.xlsx')
    return d


def plot_induced_bar(dataset='Actor', model='GraphSage', *, save_dir=None):
    y = pd.read_excel('./out/all_mean_std.xlsx', index_col=0, sheet_name='detail')
    y = y[y['dataset'] == dataset]
    y = y[y['model'] == model]
    y = y[y['sampled_ratio'] >= 0.3]
    y = y[y['sampled_ratio'] < 1.0]
    y = y[['sample', 'induced', 'sampled_ratio', 'test_acc_mean']]

    metric = pd.DataFrame()
    g = y.groupby(['sample', 'sampled_ratio'])
    for (s_, r_), index in g.groups.items():
        acc = y.loc[index]
        delta = acc[acc['induced'] == True]['test_acc_mean'].item() / acc[acc['induced'] == False][
            'test_acc_mean'].item() - 1
        metric = metric.append({'sample': s_, 'sampled_ratio': r_, 'delta': delta}, ignore_index=True)
    metric['delta'] = 100 * metric['delta']
    results = metric.rename(columns={'sampled_ratio': 'sampling ratio', 'delta': 'induced improvement (%)'})
    ax = seaborn.barplot(x='sample', y='induced improvement (%)', data=results, hue='sampling ratio')
    plt.xlabel("sampling method")
    plt.tight_layout()
    if save_dir:
        fig = ax.get_figure()
        fig.savefig(osp.join(save_dir, f'{dataset}_{model}_induced_improved.png'), dpi=800)
    else:
        plt.show()


if __name__ == '__main__':
    # rps = get_result_files(result_dir="../out/node_classification_cut_1111_3layer",
    #                        pattern="Citeseer-*-CutEdgeSampler-*2layer.xlsx")  # pattern="Actor-*-BreadthFirstSearchSampler-*"
    # pp(rps)
    # plot_box(rps, model_names=models)
    # plot_box(rps, auto_ylim=True, sampled_ratio='left_p', save_dir="../out/node_classification_cut")
    # gnn_performance = find_best2_and_worst2_gnn_by_mean(rps)
    # write_mean_std(rps, save_dir="./out/node_classification_cut", sampled_ratio='left_p')
    # t = xlxs2table3()
    rps = get_result_files(result_dir="../out/sample_label_pubmed")
    plot_label_entropy_box(rps, save_dir=None)
    # plot_induced_bar(dataset='Pubmed', model='GCN', save_dir='./out')
