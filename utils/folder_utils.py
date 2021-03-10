import os 

def check_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

def get_working_dir_name(root_dir, args):
    folders = os.listdir(root_dir)
    num = max([int(f.split('_')[1]) for f in folders if f.split('_')[0] == args.model]) + 1 
    return f'{args.model}_{str(num)}'

def make_results_folders(working_dir):
    check_dir(os.path.join(working_dir, 'weights'))
    check_dir(os.path.join(working_dir, 'test_img'))