import argparse
import numpy

if __name__=='__main__':
	'''
	parser = argparse.ArgumentParser()
	parser.add_argument("-tr", "--trainfile", type=int, nargs='+', default=[], help="Joint parallel file of two languages; sentences separated by |||")
	
	args = parser.parse_args()
	x = args.trainfile

	for val in x:
		print val
	'''

	arrays = numpy.load('model_param.npz')

	for a in arrays['arr_0']:
		print a
		print '======================'
