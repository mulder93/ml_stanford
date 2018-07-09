function weigths = randInitializeWeights(inputSize, outputSize)
	epsilonInit = 0.12;
	weigths = rand(outputSize, 1 + inputSize) * 2 * epsilonInit - epsilonInit;
end
