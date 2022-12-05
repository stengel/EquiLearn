import os
import index_algorithm
import sys

sys.path.append(os.getcwd())

def run():
    #running lrsnash
    os.system('./lrs/lrsnash lrs/lrsnash_input > lrs/lrsnash_output')
    execfile('lrs/process_lrsnash_output.py')

    #running clique
    os.system('./clique/clique < clique/clique_input > clique/clique_output')

    #running index calculation
    index_algorithm.run()

if __name__ == '__main__':
    run()
