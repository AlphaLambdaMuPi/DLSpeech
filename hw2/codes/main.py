import logging, logging.handlers
import sys
from settings import *
from datetime import datetime
import subprocess, os
import re
import requests

def main():

    record = True
    print('Enter a name (default = yapsilon) or konkon if you dont want to '
          'record')
    name = input()
    if name == 'konkon':
        record = False


    now = datetime.now()
    now_str = now.strftime('%m%d_%H%M%S')
    if name: now_str += '_' + name
    now_dir = os.path.join(OUTPUT_PATH, now_str)
    if record: os.mkdir(now_dir)
    rec_file = os.path.join(os.path.abspath(now_dir), 'w.out')


    SVM_FILE = 'svm_api'
    C = 100
    E = 200
    cmd = [SVM_LEARN_PATH, '--m', SVM_FILE, '-c', str(C), '-e', str(E),
           '_____yapsilon', rec_file]
    if not record: cmd = cmd[:-1]

    catchEps = re.compile(r'CEps=(\d+\.\d*),')

    with subprocess.Popen(cmd, stdout=subprocess.PIPE,
                          bufsize=10) as p:
        try:
            for line in p.stdout:
                s = line.decode()
                print(s)
                mt = catchEps.search(s)
                if mt:
                    requests.post('http://140.112.18.227:5000/send', 
                                 {'name': now_str, 
                                  'value': mt.group(1),
                                  'time_stamp': str(datetime.now())})
        except KeyboardInterrupt:
            p.terminate()
            os.kill(p.pid, 0)




if __name__ == '__main__':
    main()
