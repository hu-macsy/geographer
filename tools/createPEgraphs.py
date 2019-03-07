import numpy as np 
import os
from random import randint


def makeRandomTree( numLevels, firstLevel, maxChildrenPerLevel, minChildren=1,):

	assert( len(maxChildrenPerLevel)==numLevels)

	tree = list()
	tree.append( [firstLevel] )

	for i in range(1,numLevels):	# for all the levels
		prevLevel = tree[i-1] 		# a list of numbers
		newLevel = list()
		for j in prevLevel:			# for all the nodes in the previous level
			maxChildren = maxChildrenPerLevel[i]
			assert(minChildren<=maxChildren)
			# procude a random number of children for every node in the previous level
			for r in range(j):
				newLevel.append( randint(minChildren,maxChildren) )
				#TODO: maybe we do not want nodes with 1 child

		assert( sum(prevLevel)==len(newLevel) )
		tree.append( newLevel )

	return tree

#-----------------------------------------------------------------------------

def createLeafLabels(tree):
	
	numLevels = len(tree)
	labelTree = list()
	labelTree.append( [''] )

	for i in range(1,numLevels+1):	# for all the levels
		prevLevel = tree[i-1]
		#thisLevel = tree[i]
		prevLabelLvl = labelTree[i-1]
		#thisLabelLvl = labelTree[i]

		newLabelLevel = list()

		for j in range( len(prevLevel) ):
			prevLabel = prevLabelLvl[j]
			numChildren = prevLevel[j]
			for r in range(numChildren):
				newLabelLevel.append( prevLabel+","+str(r) )
			
		assert( sum(prevLevel)==len(newLabelLevel) )
		labelTree.append( newLabelLevel )

	return labelTree

#-----------------------------------------------------------------------------

def writeInFile( filename, labels, cpu, mem):

	if os.path.exists(filename):
		print("File", filename, "exists.\nAborting....")
		return False

	numPEs = len(labels) 
	assert( numPEs==len(cpu) )
	assert( numPEs==len(mem) )

	with open(filename,"w+") as f:
		f.write( "#(label), mem(GB), cpu(%)\n")
		f.write( str(numPEs)+"\n" )
		for i in range(numPEs):
			thisLabel = labels[i]
			line = thisLabel + " # " + str(mem[i])+", " + str(cpu[i]) +"\n"
			f.write(line)

	print("Data written in file", filename )
	return True

#-----------------------------------------------------------------------------
'''
tree= [ [3], 
		[2,3,4],
		[3,2, 4,1,2, 2,3,4,5]]

numPEs =  sum( tree[-1] )

mem = [64]*numPEs #all PEs have the same memory
cpu = [1]*numPEs #all PEs have the same cpu speed
'''

## create and store in a file a random tree

numLevels = 4 # actual number of levels is -1 since the first level is a ghost node 
firstLevel = 4 
maxChildren = 16
minChildren = 2 #turn also minChildren to minChildrenPerLevel?
#variable firstLevel will not adhere to maxChildrenPerLevel, 
#i.e., maxChildrenPerLevel[0] is not considered
#maxChildrenPerLevel = [maxChildren]*numLevels
maxChildrenPerLevel = [2, 3, 4, 5] 

assert( len(maxChildrenPerLevel)==numLevels)

tree2 = makeRandomTree( numLevels,firstLevel, maxChildrenPerLevel, minChildren )
assert( len(tree2)==numLevels )
numPEs2 =  sum( tree2[-1] )

mem = [64]*numPEs2 #all PEs have the same memory
cpu = [1]*numPEs2 #all PEs have the same cpu speed

percentages = [0, 0.2, 0.7, 1] #how many different groups we have and how large they are
percentages2 = [0, 0.8, 1, 0.9] #how cpu and mem differs

groupInds = [ int(perc*numPEs2) for perc in percentages ]
print(groupInds)

for g in range(1, len(groupInds) ):
	start = groupInds[g-1]
	end = groupInds[g]
	for i in range(start,end):
		mem[i] *= percentages2[g]
		cpu[i] *= percentages2[g]

finalLabels = [ label[1:] for label in createLeafLabels(tree2)[-1] ]


file = "testPEgraph" + str(numPEs2) + ".txt"
writeInFile(  file, finalLabels, cpu, mem )
