# DTA-GTOmega
The source code of DTA-GTOmega

# Environment

# Feature Generation

```commandline
git clone https://github.com/HeliXonProtein/OmegaFold
cd OmegaFold
python setup.py install
```

or our Modified version for feature extraction

use the following commandline to generate the necessary directories

```commandline
cd omegafold_feature
mkdir Davis/omega_2
mkdir Davis/pdbs
mkdir Davis/pdb_contact_map
mkdir Davis/cut_seqs
mkdir pretrained_model
```

The weight we use can be downloaded from OmegaFold Source or Our Drive
- https://pan.baidu.com/s/1Lg8WDZWl4srPkwIJZgO3cA
- with password 'CUTE'

```commandline
mv model.pt pretrained_model/model.pt
```

Then (1)Check omegafold_feature/OmegaFeatureDemo.ipynb, the notebook is friendly to generate features.
(2)Check the Davis/contact.ipynb for contact map files

Finnaly, mv the features to data

```commandline
mkdir ../data/Davis_omega
cp Davis/omega_2 ../data/Davis_omega -r
cp Davis/pdb_contact_map ../data/Davis_omega -r
```
