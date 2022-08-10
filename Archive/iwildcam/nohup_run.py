import os
import sys
os.system("nohup sh -c '" +
          sys.executable + " run.py >results/nohup/run_syn.txt ' &")
