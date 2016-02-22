import sys
import subprocess
import os

username = 'username'
email = 'user@example.com'

if len(sys.argv) > 1:
    jobname = sys.argv[1]
    
    path = os.path.join('/', 'home', 'kurse', username, 'deepnlp', 'results', jobname)
    
    if os.path.exists(path):
        print('Folder', jobname, 'already exists. Please choose another one.')
    else:
        os.makedirs(path, exist_ok=True)
        
        job = """#!/bin/bash
        
        #BSUB -J """ + jobname + """
        #BSUB -eo """ + os.path.join(path, 'stderr.txt') + """
        #BSUB -oo """ + os.path.join(path, 'stdout.txt') + """
        #BSUB -n 1
        #BSUB -M 28672 
        #BSUB -W 12:00
        #BSUB -x
        #BSUB -q kurs3
        #BSUB -u """ + email + """
        #BSUB -N
        
        cd """ + path + """
        cp ../../test3.py test3.py
        THEANO_FLAGS='mode=FAST_RUN,device=gpu,force_device=True,floatX=float32' python3 test3.py
        """
        
        args = ['bsub']
        
        proc = subprocess.Popen(args, stdin=subprocess.PIPE, universal_newlines=True)
        try:
            proc.communicate(job, timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            outs, errs = proc.communicate()