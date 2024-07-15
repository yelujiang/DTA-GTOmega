# -*- coding: utf-8 -*-
# =============================================================================
# Copyright 2022 HeliXon Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""
The main function to run the prediction
"""
# =============================================================================
# Imports
# =============================================================================
import gc
import logging
import os
import sys
import time

import torch

# import omegafold as of
# import numpy as of
from model import OmegaFold
from config import make_config
import pipeline
import pickle as pkl

# =============================================================================
# Functions
# =============================================================================

@torch.no_grad()
def getTrainedModel(input_file, output_dir):
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    args, state_dict, forward_config = pipeline.get_args(input_file=input_file,
        output_dir=output_dir)
    # return args, state_dict, forward_config
    # create the output directory
    os.makedirs(args.output_dir, exist_ok=True)
    # get the model
    logging.info(f"Constructing OmegaFold")
    model = OmegaFold(make_config(args.model))
    if state_dict is None:
        logging.warning("Inferencing without loading weight")
    else:
        if "model" in state_dict:
            state_dict = state_dict.pop("model")
        model.load_state_dict(state_dict)
    # args.device = 'cuda:1'
    model.eval()
    model.to(args.device)
    return model, args, state_dict, forward_config
    print('Load model done')

@torch.no_grad()
def main(model, input_file, output_dir, args, state_dict, forward_config):
    # logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    # args, state_dict, forward_config = pipeline.get_args(input_file=input_file,
    #     output_dir=output_dir)
    # # return args, state_dict, forward_config
    # # create the output directory
    # os.makedirs(args.output_dir, exist_ok=True)
    # # get the model
    # logging.info(f"Constructing OmegaFold")
    # model = OmegaFold(make_config(args.model))
    # if state_dict is None:
    #     logging.warning("Inferencing without loading weight")
    # else:
    #     if "model" in state_dict:
    #         state_dict = state_dict.pop("model")
    #     model.load_state_dict(state_dict)
    # model.eval()
    # model.to(args.device)
    # print('done')
    # return None
    logging.info(f"Reading {args.input_file}")
    # print('Read..')
    # return None
    with open(input_file) as f:
        lines = f.readlines()
        skip_lines = lines[0::2]
        if len(skip_lines[0].split(':'))>1:
            chains = [skip_lines[i].split(':')[1] for i in range(len(skip_lines))]
        else:
            chains = ''
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
        logging.info(f"Predicting {i + 1}th chain in {args.input_file}")
        logging.info(
            f"{len(input_data[0]['p_msa'][0])} residues in this chain."
        )
        ts = time.time()
        try:
            for k in input_data:
                for kk in k:
                    k[kk] = k[kk].to(model.device)
            output, saved_embed = model(
                    input_data,
                    predict_with_confidence=True,
                    fwd_cfg=forward_config
                )
        except RuntimeError as e:
            logging.info(f"Failed to generate {save_path} due to {e}")
            logging.info(f"Skipping...")
            continue
        logging.info(f"Finished prediction in {time.time() - ts:.2f} seconds.")

        logging.info(f"Saving prediction to {save_path}")
        pipeline.save_pdb(
            pos14=output["final_atom_positions"],
            b_factors=output["confidence"] * 100,
            sequence=input_data[0]["p_msa"][0],
            mask=input_data[0]["p_msa_mask"][0],
            save_path=save_path.split('/')[1] + '/pdbs/' + save_path.split('/')[-1],
            model=0
        )
        if chains!='':
            chain = chains[i]
        else:
            chain = ''
        f = open(f'{save_path[:-4]}{chain}.pkl','wb')
        pkl.dump(saved_embed,f)
        f.close()
        logging.info(f"Saved")
        del output
        torch.cuda.empty_cache()
        gc.collect()
    logging.info("Done!")
    # return saved_embed


# =============================================================================
# Tests
# =============================================================================
if __name__ == '__main__':
    main()
