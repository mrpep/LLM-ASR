import gin
from pathlib import Path
import torch
from ginpipe.core import gin_configure_externals
import re

def config_to_dict(config):
    with open(config,'r') as f:
        config = f.read()
    config_lines = config.split('\n')
    
    acc_val = ''
    acc_key = ''
    config_d = {}
    for l in config_lines:
        if not l.startswith('#'):
            if '=' in l:
                if acc_key != '':
                    config_d[acc_key]=acc_val
                acc_key=l.split('=')[0].strip()
                acc_val=l.split('=')[1].strip()
                if acc_val == '\\':
                    acc_val=''
            else:
                acc_val+=l.strip()
    config_d[acc_key] = acc_val
        
    return config_d

def fuzzy_get(d, key):
    gathered = []
    for k,v in d.items():
        #Caso 1: key sin nada raro:
        if ('.' not in key) and ('/' not in key):
            if k == key:
                gathered.append((k, v))
            elif ('.' in k) and (k.split('.')[0] == key):
                gathered.append((k, v))
        #Caso 2: key con puntos:
        elif ('.' in key) and ('/' not in key):
            #Subcaso 1: match perfecto
            if k == key:
                gathered.append((k,v))
            #Subcaso 2: matcheo ultimo punto con primero. Ej: pl.callbacks.LearningRateMonitor es key, LearningRateMonitor.algo es k
            elif key.split('.')[-1] == k.split('.')[0]:
                gathered.append((k,v))
            else:
                pass
        #Caso 3: key en scope + puntos:
        elif ('/' in key):
            #Aca puede pasar que si tengo train/blabla.DataLoader, quiera buscar train/Dataloader.
            if k.split('.')[0] == '{}/{}'.format(key.split('/')[0], key.split('/')[1].split('.')[-1]):
                gathered.append((k, v))
            elif k == key:
                gathered.append((k, v))
        else:
            print('mmm')
    return gathered
            
def get_target_d(d, new_d, target,):
    gathered = fuzzy_get(d, target)
    for l in gathered:
        target, val = l
        k = []
        if '@' in val:
            k += re.findall(r'@([^,\s\[\]]+)', val)
        if '%' in val:
            k += re.findall(r'%([^,\s\[\]]+)', val)
        if len(k) > 0:
            for ki in k:
                get_target_d(d, new_d, ki)
        new_d[target] = val

def get_model_config(config_path, targets, replacements, additions=None):
    d = config_to_dict(config_path)
    if additions is not None:
        for k,v in additions.items():
            d[k]=v
    pruned_config_str = ''
    pruned_config = {}
    for target in targets:
        get_target_d(d, pruned_config, target)
    for k,v in replacements.items():
        if k in pruned_config:
            pruned_config[v] = pruned_config.pop(k)
    for k,v in sorted(pruned_config.items()):
        if '.' not in k:
            pruned_config_str += '\n{}={}'.format(k,v)
    pruned_config_str+='\n'
    for k,v in sorted(pruned_config.items()):
        if '.' in k:
            pruned_config_str += '\n{}={}'.format(k,v)
    return pruned_config_str