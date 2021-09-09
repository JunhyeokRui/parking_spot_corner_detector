import os
import argparse
import yaml

parser = argparse.ArgumentParser(description='corner_annotation')
parser.add_argument('--dataset', default='AVM_center_data_4class_test')
parser.add_argument('--train', required=True)
parser.add_argument('--valid', required=True)
parser.add_argument('--test', required=True)
parser.add_argument('--run', required=True)

args = parser.parse_args()

path = os.path.join('../{}'.format(args.dataset))

train = args.train.split(',')
test = args.test.split(',')
valid = args.valid.split(',')

run = args.run

txt_path = os.path.join('data','dyros',run)
os.makedirs(txt_path,exist_ok = True)

train_txt  = open(os.path.join(txt_path,'train.txt'),'w')
test_txt  = open(os.path.join(txt_path,'test.txt'),'w')
valid_txt  = open(os.path.join(txt_path,'valid.txt'),'w')

train_txt.seek(0) 
test_txt.seek(0)
valid_txt.seek(0)

for this_trial in os.listdir(os.path.join(path,'images')):
    if 'trial' in this_trial:
        for this_image in os.listdir(os.path.join(path,'images',this_trial)):
            if 'jpeg' in this_image and os.path.isfile(os.path.join(path,'labels',this_trial,'{}.txt'.format(this_image[:-5]))):
                if this_trial[6] in train:
                    train_txt.write('../{}/images/{}/{}\n'.format(args.dataset, this_trial,this_image))
                if this_trial[6] in test:
                    test_txt.write('../{}/images/{}/{}\n'.format(args.dataset, this_trial,this_image))
                elif this_trial[6] in valid:
                    valid_txt.write('../{}/images/{}/{}\n'.format(args.dataset, this_trial,this_image))


yaml_dict = {'train': './data/dyros/{}/train.txt'.format(args.run),
          'val':'./data/dyros/{}/test.txt'.format(args.run),
          'test':'./data/dyros/{}/valid.txt'.format(args.run),
          'nc':4,
          'names': ['out','in','aux_out','aux_in']
}

with open('./data/{}.yaml'.format(args.run), 'w') as f:
    yaml.dump(yaml_dict, f, default_flow_style=None)

