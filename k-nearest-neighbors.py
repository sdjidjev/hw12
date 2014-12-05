from collections import Counter
import numpy

k = [1, 2, 5,10,25]
f = open('hw12data/digitsDataset/trainFeatures.csv', 'r')
f1 = open('hw12data/digitsDataset/trainLabels.csv', 'r')
f2 = open('hw12data/digitsDataset/valFeatures.csv', 'r')
f3 = open('hw12data/digitsDataset/valLabels.csv', 'r')
f5 = open('hw12data/digitsDataset/testFeatures.csv','r')
f6 = open('digitsOutput.csv','w')

train_features = f.read().splitlines()
train_features_number = f1.read().splitlines()
train_features = [map(lambda y: float(y), x.split(',')) for x in train_features]
train_features = zip(train_features,train_features_number)
val_features = f2.read().splitlines()
val_features = [map(lambda y: float(y), x.split(',')) for x in val_features]
val_labels = f3.read().splitlines()
test_features = f5.read().splitlines()
test_features = [map(lambda y: float(y), x.split(',')) for x in test_features]

def classify(x,k):
	distance_array = []
	for i in range(1,len(train_features)):
		distance_array.append((distance(x,train_features[i][0]),train_features[i][1]))
	return tally_votes(sorted(distance_array)[0:k])
	# return tally_votes([x[1] for x in sorted(distance_array)][0:k])

def distance(x,i):
	return numpy.linalg.norm(numpy.array(x)-numpy.array(i))

def tally_votes(distance_array):
	count = Counter([x[1] for x in distance_array])
	most = count.most_common()
	# take all classes that have equivalent tallies to the highest.
	most = [x[0] for x in most if x[1] == most[0][1]]
	# helper function that returns the first class that appears in the list.
	# Because it's sorted the first one will be the closest.
	def foo(most):
		for i in distance_array:
			if i[1] in most:
				return i
			print "poot."
	return foo(most)[1]

# f8 = open("writeup1.txt", "w")
# for i in k:
# 	counter = 0.0
# 	f4 = open("digitsOutput"+str(i)+".csv","w")
# 	for j in range(0,len(val_features)):
# 		classy =  classify(val_features[j],i)
# 		f4.write(str(classy)+"\n")
# 		if classy != val_labels[j]:
# 			counter += 1.0
# 			print "Oops: "+str(j+1)
# 	f4.close()
# 	print "For k = "+str(i)+", error rate was: "+str(counter/float(len(val_features)))
# 	f8.write("For k = "+str(i)+", error rate was: "+str(counter/float(len(val_features)))+"\n")
# f8.close()

for j in range(0,len(test_features)):
	classy =  classify(test_features[j],5)
	f6.write(str(classy)+"\n")
	print j
f6.close()
