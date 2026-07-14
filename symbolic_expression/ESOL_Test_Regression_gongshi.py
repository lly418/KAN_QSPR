# 这个笔记本包含了重建论文中所示结果的基本代码，包括特征选择、模型选择和模型评估。
import math
import os
from math import sqrt

from paretoset import paretoset
from rdkit.Chem import rdMolDescriptors  # ✅ 必须显式导入

import dgl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from dgllife.utils import smiles_to_bigraph, RandomSplitter, CanonicalAtomFeaturizer,ScaffoldSplitter
from kan import KAN
from kan.utils import ex_round
from rdkit import Chem
from sklearn.metrics import classification_report, roc_auc_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from utils import get_data, feature_selection, model_selection, plot_heatmaps, plot_pareto, plot_cm, \
    feature_selection_return, get_data_return, model_selection_return

from rdkit.Chem import rdMolDescriptors
from sklearn.model_selection import train_test_split

from rdkit.Chem import rdMolDescriptors,Descriptors
def caculate_gongshi(X):
    M = X.iloc[:, 0]
    L = X.iloc[:, 1]
    T = X.iloc[:, 2]
    result = -1.101*np.log(0.2*M+9.999)+16.1*np.log(8.703-0.405*L)+0.019*T**(0.5)-31.653
    return result

def get_X(featsdf1,df):
    # Get pareto set
    fpset1 = featsdf1[featsdf1['pareto'] == True]
    # Select points with num_feats <= 10
    under_10_pset1 = fpset1.loc[fpset1['num_feats'] <= 10]
    # Get the index of the highest F1-Score among these points
    idx1 = under_10_pset1.loc[under_10_pset1['f1_scores'] == under_10_pset1['f1_scores'].max()].index
    # 这意味着功能选择已经完成，我们正在使用以下功能：
    kept_feats1 = featsdf1.iloc[idx1]['features'].values[0].tolist()
    X_1 = df[kept_feats1]
    return X_1
# 描述符计算
def calc_all_descriptors(mol):
    # 拓扑特征扩展
    aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    ap = aromatic_rings / (mol.GetNumHeavyAtoms() + 1e-6)  # 芳香原子比例（Delaney模型核心参数）
    return [
        # 基础性质
        Descriptors.MolWt(mol),  # 分子量 74  x1
        Descriptors.MolLogP(mol),  # 脂溶性（LogP） 75 x2
        Descriptors.TPSA(mol),  # 极性表面积 76  x3
        # rdMolDescriptors.CalcTPSA(mol),  # 独立TPSA计算

        # 氢键特征
        rdMolDescriptors.CalcNumHBD(mol),  # 氢键供体 77
        rdMolDescriptors.CalcNumHBA(mol),  # 氢键受体 78

        # 拓扑特征
        rdMolDescriptors.CalcNumRotatableBonds(mol),  # 可旋转键数 79
        rdMolDescriptors.CalcNumAromaticRings(mol),  # 芳香环数量 80
        rdMolDescriptors.CalcNumHeterocycles(mol),  # 杂环数
        rdMolDescriptors.CalcNumRings(mol),  # 总环数

        # 电子特性
        Descriptors.MaxPartialCharge(mol),  # 最大部分电荷
        Descriptors.MinPartialCharge(mol),  # 最小部分电荷
        Descriptors.MaxAbsPartialCharge(mol),  # 最大绝对部分电荷
        Descriptors.MinAbsPartialCharge(mol),  # 最小绝对部分电荷

        # 功能基团计数（如羟基、羧酸基等）
        len(mol.GetSubstructMatches(Chem.MolFromSmarts('[OH]'))),  # 羟基数量 87
        len(mol.GetSubstructMatches(Chem.MolFromSmarts('C(=O)O'))),  # 羧酸基数量
        ap # 芳香原子比例（Delaney模型核心参数）
    ]

def collate_molgraphs(data):
    """Batching a list of datapoints for dataloader.

    Parameters
    ----------
    data : list of 3-tuples or 4-tuples.
        Each tuple is for a single datapoint, consisting of
        a SMILES, a DGLGraph, all-task labels and optionally
        a binary mask indicating the existence of labels.

    Returns
    -------
    smiles : list
        List of smiles
    bg : DGLGraph
        The batched DGLGraph.
    labels : Tensor of dtype float32 and shape (B, T)
        Batched datapoint labels. B is len(data) and
        T is the number of total tasks.
    masks : Tensor of dtype float32 and shape (B, T)
        Batched datapoint binary mask, indicating the
        existence of labels. If binary masks are not
        provided, return a tensor with ones.
    """
    assert len(data[0]) in [3, 4], \
        'Expect the tuple to be of length 3 or 4, got {:d}'.format(len(data[0]))
    if len(data[0]) == 3:
        smiles, graphs, labels = map(list, zip(*data))
        masks = None
    else:
        smiles, graphs, labels, masks = map(list, zip(*data))

    bg = dgl.batch(graphs)

    # print("The batched graph shape: ", len(graphs))

    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)
    # labels = torch.stack(labels, dim=1)

    labels_collate = []
    for g in labels:
        labels_collate.append(g)

    # print("The batched labels shape: ", len(labels))
    # print("The batched labels_collate shape: ", len(labels_collate))

    if masks is None:
        masks = torch.ones(labels.shape)
    else:
        masks = torch.stack(masks, dim=0)
    return smiles, bg, labels, masks
def load_dataset_for_classification(args):
    """Load dataset for classification tasks.

    Parameters
    ----------
    args : dict
        Configurations.

    Returns
    -------
    dataset
        The whole dataset.
    train_set
        Subset for training.
    val_set
        Subset for validation.
    test_set
        Subset for test.
    """
    assert args['dataset'] in ['Tox21','ClinTox','BBBP','ESOL']
    if args['dataset'] == 'Tox21':
        # from dgl.data.chem import Tox21 ## Older verson
        from dgllife.data import Tox21
        dataset = Tox21(smiles_to_bigraph, args['atom_featurizer'])
        train_set, val_set, test_set = RandomSplitter.train_val_test_split(
            dataset, frac_train=args['frac_train'], frac_val=args['frac_val'],
            frac_test=args['frac_test'], random_state=args['random_seed'])
    if args['dataset'] == 'ClinTox':
        # Import ClinTox dataset from dgllife
        from dgllife.data import ClinTox
        dataset = ClinTox(smiles_to_bigraph, args['atom_featurizer'])

        # Split the dataset into training, validation, and test sets
        train_set, val_set, test_set = RandomSplitter.train_val_test_split(
            dataset, frac_train=args['frac_train'], frac_val=args['frac_val'],
            frac_test=args['frac_test'], random_state=args['random_seed'])
    if args['dataset'] == 'BBBP':
        # Import BBBP dataset from dgllife
        from dgllife.data import BBBP
        dataset = BBBP(smiles_to_bigraph, args['atom_featurizer'])

        # Split the dataset into training, validation, and test sets
        train_set, val_set, test_set = ScaffoldSplitter.train_val_test_split(
            dataset, frac_train=args['frac_train'], frac_val=args['frac_val'],
            frac_test=args['frac_test'])
    if args['dataset'] == 'ESOL':
        from dgllife.data import ESOL
        dataset = ESOL(smiles_to_bigraph, args['atom_featurizer'])
        train_set, val_set, test_set = RandomSplitter.train_val_test_split(
            dataset, frac_train=args['frac_train'], frac_val=args['frac_val'],
            frac_test=args['frac_test'], random_state=args['random_seed'])

    return dataset, train_set, val_set, test_set
## args
args = {}
args['featsdf_is_ok'] = True
args['dataset'] = 'ESOL'
args['exp'] = 'KAN_ESOL_Add'
experimental_config = {
    'gat_hidden_feats_cca_ssg': 32,
    'random_seed': 42,
    'batch_size': 1,  ## 'batch_size' = 1 will work as Target size [1,12] is then same as input size [1,12]
    'lr': 1e-3,
    'num_epochs': 10,  ##
    'atom_data_field': 'h',
    'frac_train': 0.80,
    'frac_val': 0.10,
    'frac_test': 0.10,
    'in_feats': 79,
    'classifier_hidden_feats': 64,
    # 'classifier_hidden_feats': 128,
    'num_heads': [4, 4],
    'patience': 10,
    'atom_featurizer': CanonicalAtomFeaturizer(),
    'metric_name': 'roc_auc',
    'out_dim': 32,
    'train_batch_size': 1,
    'n_hidden':12,
    'n_layers':2,
    'n_input':79,
    'device':'cpu',
}
args.update(experimental_config)
# 利用图神经网络提取分子结构特征
args['device'] = 'cpu'
# 该设置是基于分类问题的，但可以通过整理df数据框，将其简单地扩展到故障检测或严重性分类。注释代码中提供了一个故障检测示例。
# 这个名称用于后续的模型的实验结果保存，所以注意它的名字！！！！！！！！！！！
experiment_name = 'esol_classification_add'# 这里在原有原子的特征之上加上了其他的相关元素
exp_seed = 42 # Add here the random seed 【42 2025 15 】
dataset, train_set, val_set, test_set = load_dataset_for_classification(args)
loader = DataLoader(dataset, batch_size=args['batch_size'], collate_fn=collate_molgraphs,
                    drop_last=True)  # 这里图神经网络作为上游提取特征任务，就不进行划分数据集
train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], collate_fn=collate_molgraphs, drop_last=True)
args['n_tasks'] = 1  ## 利用图神经网络提取32个潜在特征


all_hidden_feat = []  # 所有分子潜在的特征
finial_labels = []
# 对model进行训练
# 假设你有一个训练数据的 DataLoader
# train_loader 是一个迭代器，返回 (smiles, bg, labels, masks)
# 选择损失函数和优化器


num_epochs = 100
device = 'cpu'

df_masks = []
# PhSCH2CO2Me
for batch_id, batch_data in enumerate(loader):
    smiles, bg, labels, masks = batch_data
    df_masks.append(np.asarray(masks[0,0]))
    # print("Input batch_data smiles size:{}".format(len(batch_data[0])) )
    # print("Input batch_data bg size:{}".format( (batch_data[1])) )
    # print("Input batch_data label size:{}".format(batch_data[2].shape) )
    # print("Input batch_data mask size:{}".format(batch_data[3].shape) )
    bg = dgl.add_self_loop(bg)  ## Added to ward off self-loop problem
    ## allow_zero_in_degree = True
    atom_feats = bg.ndata.pop(args['atom_data_field'])
    hidden_feat = torch.mean(atom_feats,dim=0).unsqueeze(dim=0)
    # 提取极性表面积（PSA）
    mol = Chem.MolFromSmiles(smiles[0])
    extended_desc = calc_all_descriptors(mol)
    hidden_feat= torch.cat((
        hidden_feat,
        torch.tensor(extended_desc).unsqueeze(dim=0)
    ), dim=1)
    all_hidden_feat.append(hidden_feat.detach().numpy().tolist())
    # 使用torch.matmul计算二进制数对应的十进制数

    # labels: [1, 12], 将其转化为一维tensor [12,]
    labels = (labels.numpy().astype(float))[0,0]  # 只对前3个性质进行预测
    # 将tensor中的所有元素转化为字符串
    # labels_str = ''.join(labels.flatten().astype(str))
    # 计算对应的十进制数
    # decimal = int(labels_str, 2)
    finial_labels.append(labels)

    # finial_labels.append(labels.numpy().flatten().tolist()) # 12个多目标标签
# 使用列表推导式降低维度
all_hidden_feat = [item for sublist in all_hidden_feat for item in sublist]
# 创建一个 DataFrame，将这两个列表作为列
df = pd.DataFrame({'feat': all_hidden_feat, 'label': finial_labels})
# 获取每个列表的长度，假设每个列表长度相同
num_columns = len(df['feat'].iloc[0])

# 根据列表长度动态生成列名
columns = [f'feat_{i}' for i in range(num_columns)]

# 提取 'feat' 列中的列表并将其拆开
feat_df = pd.DataFrame(df['feat'].tolist(), columns=columns)
# 删除原来的 'feat' 列
df = df.drop(columns='feat')
df_masks = pd.DataFrame({'masks': df_masks})
# 合并拆分后的 DataFrame
df = pd.concat([feat_df, df, df_masks], axis=1)

# 下面开始推导公式KAN的过程
# Parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
print("device:{}".format(device))
grid_size = 5
grid_eps = 0.05
k = 3
epochs = 80
use_scaler = True
data_split = (80, 10, 10)
optim = "Adam"
exp_seed = 42
# 加载保存的pickle文件
featsdf = []

featsdf.append(pd.read_pickle(os.path.join('results', f'{experiment_name}_featsdf.pkl')))
X = []
X.append(get_X(featsdf[0],df))
labels = df['label']
masks = df['masks']
res = caculate_gongshi(X[0])
r2 = r2_score(labels, res)
rmse = torch.sqrt(torch.mean((torch.tensor(res )- torch.tensor(labels)) ** 2))
print(f"Test RMSE: {rmse:.4f}")
print(f"Test R2: {r2:.4f}")
