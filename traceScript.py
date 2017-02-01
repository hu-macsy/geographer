from subprocess import call
import os

numFramesTrace = 36
dirString = os.path.expanduser("~/mesh-sequences")
prefixes = ["hugetrace-", "hugetric-"]

for p in [60]:
	for i in range(36):
		formatString = '%02d' % i
		filename = os.path.join(dirString, "hugetrace-"+'000'+formatString+".graph")
		if not os.path.exists(filename):
			print(filename + " does not exist.")
		else:
			call(["job_submit", "-p", str(p), "-c", "p", "-t", str(15), "-m", str(int(4000)), "-N", "c", "submitscriptpass", "--graphFile="+filename, "--borderDepth=200", "--minGainForNextGlobalRound=200"])
