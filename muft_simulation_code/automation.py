import os
import multiprocessing

def run_command(command):
	print("Executing command:", command)
	os.system(command)

def multiprocess_command(commands):
	p=[]
	for command in commands:
		p.append(multiprocessing.Process(target=run_command, args=(command, )))
	for process in p:
		process.start()
	for process in p:
		process.join()

commands = []
par_process = 10

for i in range(1,par_process+1):
    cmd = 'python simulation.py --cycles 100 --state_prob 0.6 --dist_prob 0.6 --sim_no '+str(i)
    commands.append(cmd)

multiprocess_command(commands)
