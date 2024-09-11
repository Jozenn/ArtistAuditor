from diffusers import AutoPipelineForText2Image, DiffusionPipeline
import torch
import argparse
import json
import os
import shutil
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description="Model inference.")
    parser.add_argument(
        "--prior", 
        type=str,
        help="The prior model used to inference."
    )
    parser.add_argument(
        "--decoder",
        type=str,
        help="The decoder model used to inference."
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        required=True,
        default="",
        help="The prompt set file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        default="",
        help="The output directory of the image (inference result)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        default=1,
        help="Random seed."
    )
    args = parser.parse_args()
    
    return args


args = parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

####### Model
pipe_prior = DiffusionPipeline.from_pretrained(args.prior, torch_dtype=torch.float16)
prior_components = {"prior_" + k: v for k,v in pipe_prior.components.items()}
pipe = AutoPipelineForText2Image.from_pretrained(args.decoder, **prior_components, torch_dtype=torch.float16)

pipe.enable_model_cpu_offload()

####### Prompt
lines = []
metadata_file_path = args.prompt_file
with open(metadata_file_path, 'r') as file:
    for line in file:
        data = json.loads(line)
        lines.append(data)

####### Output Dir
output_dir_name = args.output_dir + f'_seed={args.seed}'
if not os.path.exists(output_dir_name):
    os.makedirs(output_dir_name)

####### Generate
for i in range(len(lines)):
    file_name = lines[i]['file_name']
    prompt = lines[i]['text']

    image = pipe(prompt=prompt).images[0]
    image.save(os.path.join(output_dir_name, file_name))

####### Split by artist
files = os.listdir(output_dir_name)

target_path = output_dir_name + '_split'
os.makedirs(target_path, exist_ok=True)

prefix_count = defaultdict(list)
for filename in files:
    prefix = filename.split('_')[0]
    prefix_count[prefix].append(filename)


for artist, artworks in prefix_count.items():
    os.makedirs(os.path.join(target_path, artist), exist_ok=True)
    os.makedirs(os.path.join(target_path, artist, 'pos'), exist_ok=True)
    for file_name in artworks:
        source = os.path.join(output_dir_name, file_name)
        target = os.path.join(target_path, artist, 'pos', file_name)
        shutil.copyfile(source, target)
        