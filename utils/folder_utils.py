import os 

def check_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

def get_working_dir(args, root_dir='runs'):
    num_folder = len(os.listdir(root_dir))
    return f'{args.model}_{str(num_folder)}'
