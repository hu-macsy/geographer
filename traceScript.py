from subprocess import call
import os

dirString = os.path.expanduser("~/WAVE/Giesse-Repart/mesh-sequences")

for i in range(7):
	formatString = '%02d' % i
	p = 16*2**i
	filename = os.path.join(dirString, "refinedtrace-"+'000'+formatString+".graph")
	if not os.path.exists(filename):
		print(filename + " does not exist.")
	else:
		call(["job_submit", "-p", str(p), "-c", "p", "-t", str(15), "-m", str(int(4000)), "-N", "c", "submitscriptpass", "--graphFile="+filename, "--borderDepth=200", "--minGainForNextGlobalRound=200", "--dimensions=2"])
