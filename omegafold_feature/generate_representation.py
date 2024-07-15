from main_c import main, getTrainedModel
import os
from tqdm import tqdm
from pdb2fasta import pdb2fasta
import pickle as pkl
import pipeline
import logging
pdb_files = '../../../data/cath/dompdb'

# 将PDB转为sequence
def pdbsfromdirs2fasta(pdb_files):
    if type(pdb_files)==str and 'cath' in pdb_files:
        if 'demo' in pdb_files:
            out_dir = '../../../data-sequence/cath_demo'
            dic_path_dir = '../../../data-path-pickle/cath_demo.pkl'
        else:
            out_dir = '../../../data-sequence/cath'
            dic_path_dir = '../../../data-path-pickle/cath.pkl'
        if not os.path.exists(out_dir):
            os.system('mkdir ' + out_dir)
        files = os.listdir(pdb_files)

    elif type(pdb_files) == dict:
        pass
    # 记录所有
    if not os.path.exists(dic_path_dir):
        dic = {}
    else:
        with open(dic_path_dir,'rb') as f:
            dic = pkl.load(f)

    for pdb in tqdm(files):
        if pdb != '2qe7G01':continue
        in_file = pdb_files + '/' + pdb

        name = pdb.split('.')[0]
        if name in dic.keys() or 'ipynb' in name or name == '':
            continue

        out_file = out_dir + '/' + name
        f = open(out_file, 'w')
        f.write(pdb2fasta(in_file))
        f.close()

        dic[name] = {}
        dic[name]['pdb_path'] = os.path.abspath(in_file)
        dic[name]['seq_path'] = os.path.abspath(out_file)
    with open(dic_path_dir,'wb') as f:
            dic = pkl.dump(dic, f)

# 将sequence转为Omega特征
def generateOmegaFeatureFromSeq(dataset, ques_set=False):
    path_pkl = f'../../../data-path-pickle/{dataset}.pkl'
    with open(path_pkl,'rb') as f:
        dic = pkl.load(f)

    output_dir = f'../features/{dataset}'
    if not os.path.exists(output_dir):
        os.system('mkdir ' + output_dir)
        
    model = None
    files = dic.keys()
    
    if ques_set:
        with open('../../../debug/question_seq.pkl','rb') as f:
            files = pkl.load(f)
    for f in tqdm(files):
        # For multiple use
        # if f != '4lc3A01':continue
        already_generated = os.listdir('../../../debug/questionseq')
        # already_generated = [already.split(':')[0] for already in already_generated]
        if f in already_generated:
            continue

        if model == None:
            model, args, state_dict, forward_config = getTrainedModel(dic[f]['seq_path'], output_dir)

        args.input_file = dic[f]['seq_path']
        main(model, dic[f]['seq_path'], output_dir, args, state_dict, forward_config)
 
        with open(f'../../../debug/questionseq/{f}','w') as ff:
            ff.write('1')

# 将OmegaFeature路径记录        
def singlePkl2Dict(dataset):
    path_pkl = f'../../../data-path-pickle/{dataset}.pkl'
    with open(path_pkl,'rb') as f:
        dic = pkl.load(f)
    output_dir = f'../features/{dataset}'
    files = dic.keys()
    model = None
    quest = ['3q34A00',]
    for f in tqdm(files):
        
        if model == None:
            model, args, state_dict, forward_config = getTrainedModel(dic[f]['seq_path'], output_dir)
        # else:
        logging.info(f"Reading {args.input_file}")
        args.input_file = dic[f]['seq_path']
        input_file = dic[f]['seq_path']
        with open(input_file) as ff:
            lines = ff.readlines()
            skip_lines = lines[0::2]
            print(input_file, skip_lines)
            chains = [skip_lines[i].split(':')[1] for i in range(len(skip_lines))]
        for i, (input_data, save_path) in enumerate(
                pipeline.fasta2inputs(
                    args.input_file,
                    num_pseudo_msa=args.num_pseudo_msa,
                    output_dir=args.output_dir,
                    device=args.device,
                    mask_rate=args.pseudo_msa_mask_rate,
                    num_cycle=args.num_cycle,
                )
            ):
            chain = chains[i]
            save_path = f'{save_path.split()[0]}_{chain}.pkl'
            dic[f]['omega_path'] = os.path.abspath(save_path)
            
        break
    with open(path_pkl,'wb') as f:
        pkl.dump(dic, f)

# pdbsfromdirs2fasta(pdb_files)
generateOmegaFeatureFromSeq('cath', ques_set=True)
# singlePkl2Dict('cath')