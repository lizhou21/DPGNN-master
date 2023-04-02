import os
import json

def ensure_dir(d,verbose = True):#确定目录是否存在，不存在则创建
    if not os.path.exists(d):
        if verbose:
            print('Directory {} do not exist;creating...'.format(d))
        os.makedirs(d)

def save_config(config, path, verbose = True):
    with open(path, 'w') as outfile:
        json.dump(config,outfile,indent=2)
    if verbose:
        print("Config saved to file {}".format(path))
    return config

class FileLogger(object):
    # 定期打开文件并对其进行写入的文件记录器
    def __init__(self,filename,header = None):
        self.filename = filename
        if os.path.exists(filename):
            os.remove(filename)
        if header is not None:
            with open(filename,'w') as out:
                print(header,file=out) ##将header写入文件out中

    def log(self, message):#写入训练记录
        with open(self.filename,'a') as out:
            print(message, file=out)


def print_config(config):
    info = "Running with the following configs:\n"
    for k,v in config.items():
        info += '\t{}:{}\n'.format(k,str(v))
    print('\n'+info+'\n')
    return