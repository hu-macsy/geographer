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
				newLabelLevel.append( prevLabel+" "+str(r) )
			
		assert( sum(prevLevel)==len(newLabelLevel) )
		labelTree.append( newLabelLevel )

	return labelTree

#-----------------------------------------------------------------------------

def writeInFile( filename, labels, weights, numWeights=5):

	if os.path.exists(filename):
		print("File", filename, "exists.\nAborting....")
		return False

	numPEs = len(labels) 
	
	#numWeights = len(weights[0]+1)/3
	
	assert( numPEs==len(weights) )

	with open(filename,"w+") as f:
		f.write( "%first line contains the number of leaves, the number of node weights\n")
		f.write( "%and a bit per weights that indicates if the weight is proportional\n"
			)
		f.write( "%e.g., if bit 4 is 0 then weight 4 is not proportional\n")
		f.write( "%(label), node weights \n")
		f.write( str(numPEs) + " " + str(numWeights)  )
		for w in range(numWeights):
			f.write( " " + str( randint(0,1)) )
		f.write("\n")

		for i in range(numPEs):
			thisLabel = labels[i]
			thisWeight = weights[i] 
			line = thisLabel + " # " + thisWeight +"\n"
			f.write(line)

	print("Data written in file", filename )
	return True

#-----------------------------------------------------------------------------

## create and store in a file a random tree

numLevels = 3 # actual number of levels is -1 since the first level is a ghost node 
firstLevel = 4 
maxChildren = 16
minChildren = 2 #turn also minChildren to minChildrenPerLevel?
#variable firstLevel will not adhere to maxChildrenPerLevel, 
#i.e., maxChildrenPerLevel[0] is not considered
#maxChildrenPerLevel = [maxChildren]*numLevels
maxChildrenPerLevel = [2, 4, 4] 

assert( len(maxChildrenPerLevel)==numLevels)

tree = makeRandomTree( numLevels,firstLevel, maxChildrenPerLevel, minChildren )
assert( len(tree)==numLevels )
numPEs =  sum( tree[-1] )

numWeights = 5
allWeights = [""]*numPEs #5 node weights

#mem = [64]*numPEs #all PEs have the same memory
#cpu = [1]*numPEs #all PEs have the same cpu speed

percentages = [0, 0.2, 0.7, 1] #how many different groups we have and how large they are
percentages2 = [0, 0.8, 1, 0.9] #how cpu and mem differs

groupInds = [ int(perc*numPEs) for perc in percentages ]
print(groupInds)

for g in range(1, len(groupInds) ):
	start = groupInds[g-1]
	end = groupInds[g]
	for i in range(start,end):
		for w in range(numWeights):
			#allWeights[i][w] *= percentages2[g]
			allWeights[i] += str( randint(10, 100) )
			allWeights[i] += " "

finalLabels = [ label[1:] for label in createLeafLabels(tree)[-1] ]

file = "testPEgraph" + str(numPEs) + ".txt"
writeInFile(  file, finalLabels, allWeights )
