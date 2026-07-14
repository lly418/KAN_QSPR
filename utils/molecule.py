import dgl.backend as F
import numpy as np
from functools import partial

from dgl import DGLGraph

try:
    from rdkit import Chem
    from rdkit.Chem import rdmolfiles, rdmolops
except ImportError:
    pass

__all__ = ['one_hot_encoding', 'BaseAtomFeaturizer', 'CanonicalAtomFeaturizer',
           'mol_to_graph', 'smile_to_bigraph', 'mol_to_bigraph',
           'smile_to_complete_graph', 'mol_to_complete_graph',
           'one_of_k_encoding_unk', 'get_atom_features', 'get_bond_features']

__all__ = ['one_hot_encoding', 'BaseAtomFeaturizer', 'CanonicalAtomFeaturizer',
           'mol_to_graph', 'smile_to_bigraph', 'mol_to_bigraph',
           'smile_to_complete_graph', 'mol_to_complete_graph']

def one_hot_encoding(x, allowable_set):
    """One-hot encoding.

    Parameters
    ----------
    x : str, int or Chem.rdchem.HybridizationType
    allowable_set : list
        The elements of the allowable_set should be of the
        same type as x.

    Returns
    -------
    list
        List of boolean values where at most one value is True.
        If the i-th value is True, then we must have
        x == allowable_set[i].
    """
    return list(map(lambda s: x == s, allowable_set))

class BaseAtomFeaturizer(object):
    """An abstract class for atom featurizers

    All atom featurizers that map a molecule to atom features should subclass it.
    All subclasses should overwrite ``_featurize_atom``, which featurizes a single
    atom and ``__call__``, which featurizes all atoms in a molecule.
    """

    def _featurize_atom(self, atom):
        return NotImplementedError

    def __call__(self, mol):
        return NotImplementedError

class CanonicalAtomFeaturizer(BaseAtomFeaturizer):
    """A default featurizer for atoms.

    The atom features include:

    * **One hot encoding of the atom type**. The supported atom types include
      ``C``, ``N``, ``O``, ``S``, ``F``, ``Si``, ``P``, ``Cl``, ``Br``, ``Mg``,
      ``Na``, ``Ca``, ``Fe``, ``As``, ``Al``, ``I``, ``B``, ``V``, ``K``, ``Tl``,
      ``Yb``, ``Sb``, ``Sn``, ``Ag``, ``Pd``, ``Co``, ``Se``, ``Ti``, ``Zn``,
      ``H``, ``Li``, ``Ge``, ``Cu``, ``Au``, ``Ni``, ``Cd``, ``In``, ``Mn``, ``Zr``,
      ``Cr``, ``Pt``, ``Hg``, ``Pb``.
    * **One hot encoding of the atom degree**. The supported possibilities
      include ``0 - 10``.
    * **One hot encoding of the number of implicit Hs on the atom**. The supported
      possibilities include ``0 - 6``.
    * **Formal charge of the atom**.
    * **Number of radical electrons of the atom**.
    * **One hot encoding of the atom hybridization**. The supported possibilities include
      ``SP``, ``SP2``, ``SP3``, ``SP3D``, ``SP3D2``.
    * **Whether the atom is aromatic**.
    * **One hot encoding of the number of total Hs on the atom**. The supported possibilities
      include ``0 - 4``.

    Parameters
    ----------
    atom_data_field : str
        Name for storing atom features in DGLGraphs, default to be 'h'.
    """

    def __init__(self, atom_data_field='h'):
        super(CanonicalAtomFeaturizer, self).__init__()
        self.atom_data_field = atom_data_field

    @property
    def feat_size(self):
        """Returns feature size"""
        return 74

    def _featurize_atom(self, atom):
        """Featurize an atom

        Parameters
        ----------
        atom : rdkit.Chem.rdchem.Atom

        Returns
        -------
        results : list
            List of feature values, including boolean values and numbers
        """
        atom_types = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br',
                      'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V',
                      'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                      'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd',
                      'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb']
        results = one_hot_encoding(atom.GetSymbol(), atom_types) + \
            one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + \
            one_hot_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
            [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
            one_hot_encoding(atom.GetHybridization(),
                             [Chem.rdchem.HybridizationType.SP,
                              Chem.rdchem.HybridizationType.SP2,
                              Chem.rdchem.HybridizationType.SP3,
                              Chem.rdchem.HybridizationType.SP3D,
                              Chem.rdchem.HybridizationType.SP3D2]) + \
            [atom.GetIsAromatic()] + \
            one_hot_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])

        return results

    def __call__(self, mol):
        """Featurize a molecule

        Parameters
        ----------
        mol : rdkit.Chem.rdchem.Mol
            RDKit molecule instance.

        Returns
        -------
        dict
            Atom features of shape (N, 74),
            where N is the number of atoms in the molecule
        """
        num_atoms = mol.GetNumAtoms()
        atom_features = []
        for i in range(num_atoms):
            atom = mol.GetAtomWithIdx(i)
            atom_features.append(self._featurize_atom(atom))
        atom_features = np.stack(atom_features)
        atom_features = F.zerocopy_from_numpy(atom_features.astype(np.float32))

        return {self.atom_data_field: atom_features}

def mol_to_graph(mol, graph_constructor, atom_featurizer, bond_featurizer):
    """Convert an RDKit molecule object into a DGLGraph and featurize for it.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule holder
    graph_constructor : callable
        Takes an RDKit molecule as input and returns a DGLGraph
    atom_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for atoms in a molecule, which can be used to update
        ndata for a DGLGraph.
    bond_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for bonds in a molecule, which can be used to update
        edata for a DGLGraph.

    Returns
    -------
    g : DGLGraph
        Converted DGLGraph for the molecule
    """
    new_order = rdmolfiles.CanonicalRankAtoms(mol)
    mol = rdmolops.RenumberAtoms(mol, new_order)
    g = graph_constructor(mol)

    if atom_featurizer is not None:
        g.ndata.update(atom_featurizer(mol))

    if bond_featurizer is not None:
        g.edata.update(bond_featurizer(mol))

    return g

def construct_bigraph_from_mol(mol, add_self_loop=False):
    """Construct a bi-directed DGLGraph with topology only for the molecule.

    The **i** th atom in the molecule, i.e. ``mol.GetAtomWithIdx(i)``, corresponds to the
    **i** th node in the returned DGLGraph.

    The **i** th bond in the molecule, i.e. ``mol.GetBondWithIdx(i)``, corresponds to the
    **(2i)**-th and **(2i+1)**-th edges in the returned DGLGraph. The **(2i)**-th and
    **(2i+1)**-th edges will be separately from **u** to **v** and **v** to **u**, where
    **u** is ``bond.GetBeginAtomIdx()`` and **v** is ``bond.GetEndAtomIdx()``.

    If self loops are added, the last **n** edges will separately be self loops for
    atoms ``0, 1, ..., n-1``.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule holder
    add_self_loop : bool
        Whether to add self loops in DGLGraphs.

    Returns
    -------
    g : DGLGraph
        Empty bigraph topology of the molecule
    """
    g = DGLGraph()

    # Add nodes
    num_atoms = mol.GetNumAtoms()
    g.add_nodes(num_atoms)

    # Add edges
    src_list = []
    dst_list = []
    num_bonds = mol.GetNumBonds()
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        src_list.extend([u, v])
        dst_list.extend([v, u])
    g.add_edges(src_list, dst_list)

    if add_self_loop:
        nodes = g.nodes()
        g.add_edges(nodes, nodes)

    return g

def mol_to_bigraph(mol, add_self_loop=False,
                   atom_featurizer=CanonicalAtomFeaturizer(),
                   bond_featurizer=None):
    """Convert an RDKit molecule object into a bi-directed DGLGraph and featurize for it.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule holder
    add_self_loop : bool
        Whether to add self loops in DGLGraphs.
    atom_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for atoms in a molecule, which can be used to update
        ndata for a DGLGraph. Default to CanonicalAtomFeaturizer().
    bond_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for bonds in a molecule, which can be used to update
        edata for a DGLGraph.

    Returns
    -------
    g : DGLGraph
        Bi-directed DGLGraph for the molecule
    """
    return mol_to_graph(mol, partial(construct_bigraph_from_mol, add_self_loop=add_self_loop),
                        atom_featurizer, bond_featurizer)

def smile_to_bigraph(smile, add_self_loop=False,
                     atom_featurizer=CanonicalAtomFeaturizer(),
                     bond_featurizer=None):
    """Convert a SMILES into a bi-directed DGLGraph and featurize for it.

    Parameters
    ----------
    smile : str
        String of SMILES
    add_self_loop : bool
        Whether to add self loops in DGLGraphs.
    atom_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for atoms in a molecule, which can be used to update
        ndata for a DGLGraph. Default to CanonicalAtomFeaturizer().
    bond_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for bonds in a molecule, which can be used to update
        edata for a DGLGraph.

    Returns
    -------
    g : DGLGraph
        Bi-directed DGLGraph for the molecule
    """
    mol = Chem.MolFromSmiles(smile)
    return mol_to_bigraph(mol, add_self_loop, atom_featurizer, bond_featurizer)

def construct_complete_graph_from_mol(mol, add_self_loop=False):
    """Construct a complete graph with topology only for the molecule

    The **i** th atom in the molecule, i.e. ``mol.GetAtomWithIdx(i)``, corresponds to the
    **i** th node in the returned DGLGraph.

    The edges are in the order of (0, 0), (1, 0), (2, 0), ... (0, 1), (1, 1), (2, 1), ...
    If self loops are not created, we will not have (0, 0), (1, 1), ...

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule holder
    add_self_loop : bool
        Whether to add self loops in DGLGraphs.

    Returns
    -------
    g : DGLGraph
        Empty complete graph topology of the molecule
    """
    g = DGLGraph()
    num_atoms = mol.GetNumAtoms()
    g.add_nodes(num_atoms)

    if add_self_loop:
        g.add_edges(
            [i for i in range(num_atoms) for j in range(num_atoms)],
            [j for i in range(num_atoms) for j in range(num_atoms)])
    else:
        g.add_edges(
            [i for i in range(num_atoms) for j in range(num_atoms - 1)], [
                j for i in range(num_atoms)
                for j in range(num_atoms) if i != j
            ])

    return g

def mol_to_complete_graph(mol, add_self_loop=False,
                          atom_featurizer=None,
                          bond_featurizer=None):
    """Convert an RDKit molecule into a complete DGLGraph and featurize for it.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule holder
    add_self_loop : bool
        Whether to add self loops in DGLGraphs.
    atom_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for atoms in a molecule, which can be used to update
        ndata for a DGLGraph. Default to CanonicalAtomFeaturizer().
    bond_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for bonds in a molecule, which can be used to update
        edata for a DGLGraph.

    Returns
    -------
    g : DGLGraph
        Complete DGLGraph for the molecule
    """
    return mol_to_graph(mol, partial(construct_complete_graph_from_mol, add_self_loop=add_self_loop),
                        atom_featurizer, bond_featurizer)

def smile_to_complete_graph(smile, add_self_loop=False,
                            atom_featurizer=None,
                            bond_featurizer=None):
    """Convert a SMILES into a complete DGLGraph and featurize for it.

    Parameters
    ----------
    smile : str
        String of SMILES
    add_self_loop : bool
        Whether to add self loops in DGLGraphs.
    atom_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for atoms in a molecule, which can be used to update
        ndata for a DGLGraph. Default to CanonicalAtomFeaturizer().
    bond_featurizer : callable, rdkit.Chem.rdchem.Mol -> dict
        Featurization for bonds in a molecule, which can be used to update
        edata for a DGLGraph.

    Returns
    -------
    g : DGLGraph
        Complete DGLGraph for the molecule
    """
    mol = Chem.MolFromSmiles(smile)
    return mol_to_complete_graph(mol, add_self_loop, atom_featurizer, bond_featurizer)
# 辅助函数
def one_of_k_encoding_unk(x, allowable_set):
    '将x与allowable_set逐个比较，相同为True， 不同为False, 都不同则认为是最后一个相同'
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def get_atom_features(atom):
    # 定义一个可能的原子符号列表，其中 'DU' 代表其他未知的原子类型
    possible_atom = [
        'C', 'N', 'O', 'F', 'P', 'Cl', 'Br', 'I', 'S',  # 常见元素
        'Si', 'B', 'Na', 'Mg', 'Al', 'K', 'Ca', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Se', 'Mo', 'Ag', 'Sn', 'Ba',
        'DU',  # 添加更多元素
    ]

    # 使用 one_of_k_encoding_unk 对原子符号进行独热编码。atom.GetSymbol() 返回原子的化学符号，
    # `possible_atom` 列表中包含了可能的符号。这个方法将返回一个表示原子符号的独热编码向量。
    atom_features = one_of_k_encoding_unk(atom.GetSymbol(), possible_atom)

    # 对原子的隐性价电子数进行独热编码。atom.GetImplicitValence() 返回该原子的隐性价电子数，
    # 该数值在此仅为 0 或 1，表示有无隐性价电子。
    atom_features += one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1])

    # 对原子的自由电子数进行独热编码。atom.GetNumRadicalElectrons() 返回原子上自由电子的数量。
    # 该数值可以为 0 或 1，表示原子是否有自由电子。
    atom_features += one_of_k_encoding_unk(atom.GetNumRadicalElectrons(), [0, 1])

    # 对原子的连接度进行独热编码。atom.GetDegree() 返回原子连接的化学键的数量。
    # 这里假设连接度的最大值是 6，因此将连接度的所有可能值（0 到 6）传递给编码函数。
    atom_features += one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6])

    # 对原子的形式电荷进行独热编码。atom.GetFormalCharge() 返回原子的形式电荷，可能的值是 -1 或 1。
    atom_features += one_of_k_encoding_unk(atom.GetFormalCharge(), [-1, 1])

    # 对原子的杂化类型进行独热编码。atom.GetHybridization() 返回原子的杂化类型，
    # 这里考虑了四种类型：SP, SP2, SP3 和 SP3D。返回的值来自 Chem.rdchem.HybridizationType 枚举。
    atom_features += one_of_k_encoding_unk(atom.GetHybridization(),
                                           [Chem.rdchem.HybridizationType.SP,
                                            Chem.rdchem.HybridizationType.SP2,
                                            Chem.rdchem.HybridizationType.SP3,
                                            Chem.rdchem.HybridizationType.SP3D])

    # 将所有提取的特征组合成一个 NumPy 数组并返回
    return np.array(atom_features)


def get_bond_features(bond):
    bond_type = bond.GetBondType()
    bond_feats = [
        bond_type == Chem.rdchem.BondType.SINGLE, bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE, bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    return np.array(bond_feats)