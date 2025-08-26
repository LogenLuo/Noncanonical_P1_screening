import os
import sys
import numpy as np
import pandas as pd
import math
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
from rdkit.Chem import Descriptors
from rdkit.Chem import Draw
from rdkit import rdBase
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
suppl_enamine = Chem.SDMolSupplier('Mpro-VS-enamine.sdf')
len(suppl_enamine)
mols_enamine = [x for x in suppl_enamine if x is not None]
len(mols_enamine)
np.random.seed(1234)
np.random.shuffle(mols_enamine)
len(mols_enamine)
fp_enamine = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) for m in mols_enamine]
len(fp_enamine)
dist_matrix_enamine = [DataStructs.BulkTanimotoSimilarity(fp_enamine[i], fp_enamine[:len(fp_enamine)], returnDistance=True) for i in range(len(fp_enamine))]
len(dist_matrix_enamine)
dist_array_enamine = np.array(dist_matrix_enamine)
len(dist_array_enamine)
ward = AgglomerativeClustering(n_clusters=10)
ward.fit(dist_array_enamine)
pd.value_counts(ward.labels_)
ward_lib = {i: [] for i in range(10)}
for n, j in enumerate(ward.labels_):
    ward_lib[j].append(mols_enamine[n])
len(ward_lib)
writer = Chem.SDWriter('cluster_enamine_10_repre.sdf')
writer.SetProps(['id_number', 'dG'])
for i in range(len(ward_lib)):
    dG = list()
    for j in range(len(ward_lib[i])):
        dG.append(float(ward_lib[i][j].GetProp('r_psp_MMGBSA_dG_Bind')))
    for j in range(len(ward_lib[i])):
        dg = float(ward_lib[i][j].GetProp('r_psp_MMGBSA_dG_Bind'))
        if dg == min(dG):
            idn = ward_lib[i][j].GetProp('idnumber')
            dg = float(ward_lib[i][j].GetProp('r_psp_MMGBSA_dG_Bind'))
            ward_lib[i][j].SetProp('id_number', '{}'.format(idn))
            ward_lib[i][j].SetProp('dG', '%.2f' %(dg))
            writer.write(ward_lib[i][j])
            print(i)
            print(j)
writer.close()
pca = PCA(n_components=2)
pca.fit(dist_array_enamine)
dist_pca_enamine = pca.transform(dist_array_enamine)
np.savetxt('cluster_enamine_10_pca.csv', dist_pca_enamine, delimiter=',')
suppl_repre = Chem.SDMolSupplier('cluster_enamine_10_repre.sdf')
mols_repre = [x for x in suppl_repre if x is not None]
len(mols_repre)
print(mols_repre[i].GetProp('id_number') for i in range(len(mols_repre)))
dist_pca_repre = list()
for i in range(len(mols_repre)):
    idn = mols_repre[i].GetProp('id_number')
    for j in range(len(mols_enamine)):
        if idn == mols_enamine[j].GetProp('idnumber'):
            dist_pca_repre.append(dist_pca_enamine[j])
dist_pca_repre = np.array(dist_pca_repre)
print(dist_pca_repre)
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111)
ax.scatter(dist_pca_enamine[:, 0], dist_pca_enamine[:, 1], s=50, c=ward.labels_[:len(mols_enamine)], cmap='Paired', alpha=0.5)
ax.scatter(dist_pca_repre[:, 0], dist_pca_repre[:, 1], s=100, c=[x for x in range(len(mols_repre))], cmap='Paired', edgecolors='black')
ax.set_xlabel('PC1', fontsize=20)
ax.set_ylabel('PC2', fontsize=20)
plt.savefig('chemspace_cluster_enamine.pdf', dpi=800, transparent=True)
plt.savefig('chemspace_cluster_enamine.svg', dpi=800, transparent=True)
plt.savefig('chemspace_cluster_enamine.png', dpi=800, transparent=True) #This is Figure 1B
pattern_pyridine = AllChem.MolFromSmiles('N1=CC=CC=C1')
match_pyridine = [m for m in mols_enamine if m.HasSubstructMatch(parttern_pyridine)]
pattern_isoquinoline = AllChem.MolFromSmiles('C12=CC=NC=C1C=CC=C2')
match_isoquinoline = [m for m in mols_enamine if m.HasSubstructMatch(parttern_isoquinoline)]
pattern_pyrimidine = AllChem.MolFromSmiles('N1=CC=CN=C1')
match_pyrimidine = [m for m in mols_enamine if m.HasSubstructMatch(parttern_pyrimidine)]
pattern_17naphthyridine = AllChem.MolFromSmiles('C12=CC=NC=C1N=CC=C2')
match_17naphthyridine = [m for m in mols_enamine if m.HasSubstructMatch(parttern_17naphthyridine)]
pattern_16naphthyridine = AllChem.MolFromSmiles('C12=CC=NC=C1C=CC=N2')
match_16naphthyridine = [m for m in mols_enamine if m.HasSubstructMatch(parttern_16naphthyridine)]
pattern_5azaindole = AllChem.MolFromSmiles('C12=CC=NC=C1C=CN2')
match_5azaindole = [m for m in mols_enamine if m.HasSubstructMatch(parttern_5azaindole)]
pattern_pyridopyrazine = AllChem.MolFromSmiles('C12=CC=NC=C1N=CC=N2')
match_pyridopyrazine = [m for m in mols_enamine if m.HasSubstructMatch(parttern_pyridopyrazine)]
pattern_imidazopyridine = AllChem.MolFromSmiles('C12=CC=NC=C1NC=N2')
match_imidazopyridine = [m for m in mols_enamine if m.HasSubstructMatch(parttern_imidazopyridine)]
pca_pyridine = list()
for i in range(len(match_pyridine)):
    idn = match_pyridine[i].GetProp('idnumber')
    for j in range(len(mols_enamine)):
        if idn == mols_enamine[j].GetProp('idnumber'):
            pca_pyridine.append(dist_pca_enamine[j])
len(pca_pyridine)
pca_pyridine = np.array(pca_pyridine)
pca_pyrimidine = list()
for i in range(len(match_pyrimidine)):
    idn = match_pyrimidine[i].GetProp('idnumber')
    for j in range(len(mols_enamine)):
        if idn == mols_enamine[j].GetProp('idnumber'):
            pca_pyrimidine.append(dist_pca_enamine[j])
len(pca_pyrimidine)
pca_pyrimidine = np.array(pca_pyrimidine)
pca_isoquinoline = list()
for i in range(len(match_isoquinoline)):
    idn = match_isoquinoline[i].GetProp('idnumber')
    for j in range(len(mols_enamine)):
        if idn == mols_enamine[j].GetProp('idnumber'):
            pca_isoquinoline.append(dist_pca_enamine[j])
len(pca_isoquinoline)
pca_isoquinoline = np.array(pca_isoquinoline)
pca_17naphthyridine = list()
for i in range(len(match_17naphthyridine)):
    idn = match_17naphthyridine[i].GetProp('idnumber')
    for j in range(len(mols_enamine)):
        if idn == mols_enamine[j].GetProp('idnumber'):
            pca_17naphthyridine.append(dist_pca_enamine[j])
len(pca_17naphthyridine)
pca_17naphthyridine = np.array(pca_17naphthyridine)
pca_16naphthyridine = list()
for i in range(len(match_16naphthyridine)):
    idn = match_16naphthyridine[i].GetProp('idnumber')
    for j in range(len(mols_enamine)):
        if idn == mols_enamine[j].GetProp('idnumber'):
            pca_16naphthyridine.append(dist_pca_enamine[j])
len(pca_16naphthyridine)
pca_16naphthyridine = np.array(pca_16naphthyridine)
pca_5azaindole = list()
for i in range(len(match_5azaindole)):
    idn = match_5azaindole[i].GetProp('idnumber')
    for j in range(len(mols_enamine)):
        if idn == mols_enamine[j].GetProp('idnumber'):
            pca_5azaindole.append(dist_pca_enamine[j])
len(pca_16naphthyridine)
pca_5azaindole = np.array(pca_5azaindole)
pca_imidazopyridine = list()
for i in range(len(match_imidazopyridine)):
    idn = match_imidazopyridine[i].GetProp('idnumber')
    for j in range(len(mols_enamine)):
        if idn == mols_enamine[j].GetProp('idnumber'):
            pca_imidazopyridine.append(dist_pca_enamine[j])
len(pca_imidazopyridine)
pca_imidazopyridine = np.array(pca_imidazopyridine)
pca_pyridopyrazine = list()
for i in range(len(match_pyridopyrazine)):
    idn = match_pyridopyrazine[i].GetProp('idnumber')
    for j in range(len(mols_enamine)):
        if idn == mols_enamine[j].GetProp('idnumber'):
            pca_pyridopyrazine.append(dist_pca_enamine[j])
len(pca_pyridopyrazine)
pca_pyridopyrazine = np.array(pca_pyridopyrazine)
substruct_repre = list()
for i in range(len(mols_enamine)):
    idn = mols_enamine[i].GetProp('idnumber')
    if id == 'Z5442551082':
        substruct_repre.append(dist_pca_enamine[i])
    if id == 'Z4921678706':
        substruct_repre.append(dist_pca_enamine[i])
    if id == 'Z4921678715':
        substruct_repre.append(dist_pca_enamine[i])
    if id == 'Z5169210063':
        substruct_repre.append(dist_pca_enamine[i])
    if id == 'Z4927221500':
        substruct_repre.append(dist_pca_enamine[i])
    if id == 'Z5098613098':
        substruct_repre.append(dist_pca_enamine[i])
print(substruct_repre)
substruct_repre = np.array(substruct_repre)
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111)
ax = sns.kdeplot(x=dist_pca_enamine[:, 0], y=dist_pca_enamine[:, 1], cmap='Blues', fill=True, bw_adjust=.5, thresh=0.01)
ax.scatter(pca_pyridine[:, 0], pca_pyridine[:, 1], s=50, c='salmon', alpha=0.5, edgecolors='black', linewidths=0.5)
ax.scatter(pca_pyrimidine[:, 0], pca_pyrimidine[:, 1], s=50, c='lavender', alpha=0.5, edgecolors='black', linewidths=0.5)
ax.scatter(pca_pyridazine[:, 0], pca_pyridazine[:, 1], s=50, c='lightgreen', edgecolors='black', linewidths=0.5)
ax.scatter(pca_isoquinoline[:, 0], pca_isoquinoline[:, 1], s=50, c='red', edgecolors='black', linewidths=0.5)
ax.scatter(pca_imidazopyridine[:, 0], pca_imidazopyridine[:, 1], s=50, c='magenta', edgecolors='black', linewidths=0.5)
ax.scatter(pca_17naphthyridine[:, 0], pca_17naphthyridine[:, 1], s=50, c='purple', edgecolors='black', linewidths=0.5)
ax.scatter(pca_16naphthyridine[:, 0], pca_16naphthyridine[:, 1], s=50, c='blue', edgecolors='black', linewidths=0.5)
ax.scatter(pca_5azaindole[:, 0], pca_5azaindole[:, 1], s=50, c='brown', edgecolors='black', linewidths=0.5)
ax.scatter(substruct_repre[0, 0], substruct_repre[0, 1], s=150, c='salmon', edgecolors='black', linewidths=1.5)
ax.scatter(substruct_repre[1, 0], substruct_repre[1, 1], s=150, c='lavender', edgecolors='black', linewidths=1.5)
ax.scatter(substruct_repre[2, 0], substruct_repre[2, 1], s=150, c='red', edgecolors='black', linewidths=1.5)
ax.scatter(substruct_repre[3, 0], substruct_repre[3, 1], s=150, c='red', edgecolors='black', linewidths=1.5)
ax.scatter(substruct_repre[4, 0], substruct_repre[4, 1], s=150, c='magenta', edgecolors='black', linewidths=1.5)
ax.scatter(substruct_repre[5, 0], substruct_repre[5, 1], s=150, c='purple', edgecolors='black', linewidths=1.5)
ax.set_xlabel('PC1', fontsize=20)
ax.set_ylabel('PC2', fontsize=20)
plt.savefig('chemicalspace_substructs.pdf', dpi=800, transparent=True)
plt.savefig('chemicalspace_substructs.svg', dpi=800, transparent=True)
plt.savefig('chemicalspace_substructs.png', dpi=800, transparent=True) #This is Figure 1C
exit()





