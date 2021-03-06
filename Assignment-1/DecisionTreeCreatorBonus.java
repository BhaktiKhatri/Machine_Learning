package ml.decisiontree.id3.impl;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.*;

/**
 * Created by hemac on 9/19/2016.
 */
public class DecisionTreeCreatorBonus
{

    static Random random = new Random();

    /**
     *
     * @param arr
     * @return
     */
    static double[] computeProbability(int[] arr)
    {
        double[] noOf0sAnd1s = getNoOf0sAnd1s(arr);
        //computing probability
        return new double[]{noOf0sAnd1s[0]/arr.length, noOf0sAnd1s[1]/arr.length};
    }


    /**
     *
     * @param arr
     * @return
     */
    static double[] getNoOf0sAnd1s(int[] arr)
    {
        int noOf0s = 0;
        int noOf1s = 0;
        for (int i : arr)
        {
            if (i == 0)
                noOf0s++;
            else
                noOf1s++;
        }

        return new double[]{noOf0s, noOf1s};
    }


    /**
     *
     * @param probabilities
     * @return
     */
    static double computeEntropy(double[] probabilities)
    {
        //compute entropy using entropy formula when probabilities are known
        double entropy = 0;

        for (double probability : probabilities)
        {
            if (probability == 0 || probability == 1)
                continue;
            entropy += -(probability * ( Math.log(probability) / Math.log(2)));
        }

        return entropy;

    }


    public static void main(String[] args) throws Exception
    {
        if (args.length != 2)
        {
            System.out.println("Please enter 2 arguments as follows :: TRAINING_DATA_LOCATION TEST_DATA_LOCATION");
            return;
        }
        System.out.println(" " + args[0] + "  " + args[1]);
        String trainingDataLocation = args[0];
        String testDataLocation = args[1];
       // float pruningFactor = Float.parseFloat(args[2]);

        Input trainingInput = getAttributesValuesAndOutputs(trainingDataLocation);
        Input testInput = getAttributesValuesAndOutputs(testDataLocation);


        /**
         * 1. Train decision tree with training data using ID3 algorithm
         * 2. Compute Accuracy with training and test data
         * 3. Prune nodes of size (pruning factor * total number of nodes) randomly from the tree
         * 4. Compute Accuracy with training and test data on the pruned tree
         * 5. Display results
         */
        trainDecisionTreeAndDisplayAccuracy(trainingInput, testInput, "ID3", 0);

        for (int i = 0; i < 5; i++)
        {
            System.out.println("Iteration  ::  "+(i+1));
            trainDecisionTreeAndDisplayAccuracy(trainingInput, testInput, "RANDOMSELECTION", i+1);
        }

    }

    private static void trainDecisionTreeAlmostPrecisely(Input trainingInput, Input testInput) throws Exception
    {
        //Split training data with 80:20 ratio as training data and validation data
        TrainingAndValidationInput trainingAndValidationInput = divideTrainingDataIntoTrainingAndValidationData(trainingInput);

        System.out.println("******************** PRE PRUNED ACCURACY ********************");

        //Model a decision tree with training data
        Output output = trainAndGetDecisionTree(trainingAndValidationInput.getTrainingInput(), "");

        //Calculate accuracy of training data on decision tree
        calculateAccuracyPrintResults(output, "TRAINING DATA", 0);

        //Calculate accuracy of validation data on decision tree
        calculateAccuracyPrintResults(new Output(trainingAndValidationInput.getValidationInput(), output.getDecisionTree()), "VALIDATION DATA", 0);

        //Calculate accuracy of test data on decision tree
        calculateAccuracyPrintResults(new Output(testInput, output.getDecisionTree()), "TEST DATA", 0);
        System.out.println("*******************************************************************************");


        System.out.println("******************** PRUNED ACCURACY ********************");

        //Prune nodes using reduced error pruning method by taking inputs from validation data
        Tree prunedTree = pruneNodesUsingReducedErrorPruning(output.getDecisionTree(), trainingAndValidationInput.getValidationInput());

        //Calculate accuracy of training data on decision tree
        calculateAccuracyPrintResults(new Output(trainingAndValidationInput.getTrainingInput(), prunedTree), "TRAINING DATA", 0);

        //Calculate accuracy of validation data on decision tree
        calculateAccuracyPrintResults(new Output(trainingAndValidationInput.getValidationInput(), prunedTree), "VALIDATION DATA", 0);

        //Calculate accuracy of test data on decision tree
        calculateAccuracyPrintResults(new Output(testInput, prunedTree), "TEST DATA", 0);
        System.out.println("*******************************************************************************");
    }

    /**
     *
     * @param trainingInput
     * @return
     */
    private static TrainingAndValidationInput divideTrainingDataIntoTrainingAndValidationData(Input trainingInput)
    {
        int[][] attributesAndValues = trainingInput.getAttributesAndValues();
        String[] attributeNames = trainingInput.getAttributeNames();
        int[] outputValues = trainingInput.getOutputValues();


        int trainingDataColNumbers = (int)Math.floor(attributesAndValues[0].length * 0.8);
        int[][] attributeValuesForTraining = new int[attributesAndValues.length][trainingDataColNumbers];
        int[][] attributeValuesForValidation = new int[attributesAndValues.length][(attributesAndValues[0].length - trainingDataColNumbers)];
        int[] outPutValuesForTraining = new int[trainingDataColNumbers];
        int[] outPutValuesForValidation = new int[(attributesAndValues[0].length - trainingDataColNumbers)];

        for (int i = 0; i < attributesAndValues.length; i++)
        {
            for (int j = 0; j < trainingDataColNumbers; j++)
            {
                attributeValuesForTraining[i][j] = attributesAndValues[i][j];
                outPutValuesForTraining[j] = outputValues[j];
            }
            int col = 0;
            for (int k = trainingDataColNumbers + 1; k < attributesAndValues[0].length; k++)
            {
                attributeValuesForValidation[i][col] = attributesAndValues[i][k];
                outPutValuesForValidation[col] = outputValues[k];
                col++;
            }

        }

        Input newTrainingInput = new Input(attributeValuesForTraining, outPutValuesForTraining, attributeNames);
        Input validationInput = new Input(attributeValuesForValidation, outPutValuesForValidation, attributeNames);
        return new TrainingAndValidationInput(newTrainingInput, validationInput);
    }


    /**
     *
     * @param trainingDataLocation
     * @param testDataInput
     * @param algorithm
     * @throws Exception
     */
    public static void trainDecisionTreeAndDisplayAccuracy(Input trainingDataLocation, Input testDataInput, String algorithm, int i) throws Exception
    {
        System.out.println("******************** TRAINING USING "+algorithm+"********************");

        //Using ID3 algorithm, create a model of decision tree by taking training data as an input
        Output trainingOutput = trainAndGetDecisionTree(trainingDataLocation, algorithm);

        //Calculate accuracy of training data on decision tree
        calculateAccuracyPrintResults(trainingOutput, "TRAINING DATA", i);

        Output testOutput = new Output(testDataInput, trainingOutput.getDecisionTree());

        //Calculate accuracy of test data on decision tree
        calculateAccuracyPrintResults(testOutput, "TEST DATA", i);

        System.out.println("*******************************************************************************");


//        System.out.println("******************** PRUNED ACCURACY ********************");
//
//        Prune nodes randomly using the pruning factor
//        Tree prunedTree = pruneNodesRandomly(trainingOutput.decisionTree, (int) Math.floor(pruningFactor * getNodeCount(trainingOutput.decisionTree)));
//
        //Calculate accuracy of training data on decision tree
//        calculateAccuracyPrintResults(new Output(trainingOutput.input, prunedTree), "TRAINING DATA");

        //Calculate accuracy of test data on decision tree
//        calculateAccuracyPrintResults(new Output(testOutput.input, prunedTree), "TEST DATA");

//        System.out.println("*******************************************************************************");
    }

    /**
     *
     * @param decisionTree
     * @param validationInput
     * @return
     */
    private static Tree pruneNodesUsingReducedErrorPruning(Tree decisionTree, Input validationInput)
    {
        Tree newTree = new Tree(decisionTree);
        cloneTree(decisionTree, newTree);

        List<Tree> allLeafNodes = getAllLeafNodes(newTree);

        Set<Tree> parentsOfLeaves = new HashSet<>();

        for (Tree allLeafNode : allLeafNodes)
        {
            parentsOfLeaves.add(allLeafNode.getParent());
        }

        double accuracy = calculateAccuracy(decisionTree, validationInput);

        for (Tree parentsOfLeaf : parentsOfLeaves)
        {
            Map<String, Tree> children = parentsOfLeaf.getChildren();
            parentsOfLeaf.removeChildren();

            double newAccuracy = calculateAccuracy(newTree, validationInput);

            if (newAccuracy > accuracy)
            {
                continue;
            }

            parentsOfLeaf.setChildren(children);

        }


        return newTree;
    }

    /**
     *
     * @param decisionTree
     * @param noOfNodesToBePruned
     * @return
     */
    private static Tree pruneNodesRandomly(Tree decisionTree, int noOfNodesToBePruned)
    {

        Tree newTree = new Tree(decisionTree);
        cloneTree(decisionTree, newTree);

        List<Tree> allLeafNodes = getAllLeafNodes(newTree);

        Set<Integer> indicesToBeRemoved = new HashSet<>();

        int leafCount = getLeafCount(newTree);
        Random random = new Random();

        while (indicesToBeRemoved.size() != noOfNodesToBePruned)
        {
            indicesToBeRemoved.add(random.nextInt(leafCount));
        }


        for (Integer indexToBeRemoved : indicesToBeRemoved)
        {
            Tree tree = allLeafNodes.get(indexToBeRemoved);
            tree.getParent().removeChild(tree.getFrom());
        }

        System.out.println("No of nodes to be pruned " + noOfNodesToBePruned);
//
        System.out.println("Number of Nodes After pruning "+ getNodeCount(newTree) + "  Before Pruning " + getNodeCount(decisionTree));
//
//        System.out.println("Number of  Leaves After pruning "+ getLeafCount(newTree) + " Before Pruning " + getLeafCount(decisionTree));
//
//        System.out.println("********************************************");




        return newTree;
    }

    /**
     *
     * @param decisionTree
     * @param newTree
     */
    private static void cloneTree(Tree decisionTree, Tree newTree)
    {
        for (Map.Entry<String, Tree> stringTreeEntry : decisionTree.getChildren().entrySet())
        {
            Tree value = stringTreeEntry.getValue();
            Tree newChild  = new Tree(value);
            newTree.addChild(newChild, stringTreeEntry.getKey());
            cloneTree(value, newChild);
        }

    }

    /**
     *
     * @param trainingInput
     * @param algorithm
     * @return
     * @throws Exception
     */
    private static Output trainAndGetDecisionTree(Input trainingInput, String algorithm) throws Exception
    {
        Tree decisionTree = new Tree();
        constructDecisionTree(trainingInput.getAttributesAndValues(), trainingInput.getOutputValues(), trainingInput.getAttributeNames(), decisionTree, null, algorithm);
        return new Output(trainingInput, decisionTree);
    }

    /**
     *
     * @param traingDataLocation
     * @return
     * @throws Exception
     */
    private static Input getAttributesValuesAndOutputs(String traingDataLocation) throws Exception
    {


        BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(new FileInputStream(new File(traingDataLocation))));

        String attributeNamesRow = bufferedReader.readLine();

        String[] split = attributeNamesRow.split("\t");

        String[] attributeNames = new String[split.length - 1];

        for (int i = 0; i < split.length - 1; i++)
        {
            attributeNames[i] = split[i];
        }

        Map<Integer,List<Integer>>  attributesAndValues = new LinkedHashMap<>();

        List<Integer> output = new ArrayList<>();

        String next = bufferedReader.readLine();
        while (next != null)
        {
            if (next.isEmpty())
                continue;

            String[] split1 = next.split("\t");
            output.add(Integer.parseInt(split1[split1.length - 1]));

            for (int i = 0; i < split1.length - 1; i++)
            {
                String input = split1[i].trim();
                if (input.isEmpty())
                    continue;
                List<Integer> integers = attributesAndValues.get(i);
                if (integers == null)
                {
                    integers = new ArrayList<>();
                    attributesAndValues.put(i, integers);
                }
                integers.add(Integer.parseInt(split1[i]));
            }
            next = bufferedReader.readLine();
        }

        int[][] attributes = new int[attributesAndValues.keySet().size()][attributesAndValues.get(0).size()];

        int row = 0;
        for (Map.Entry<Integer, List<Integer>> integerListEntry : attributesAndValues.entrySet())
        {
            List<Integer> value = integerListEntry.getValue();
            int col = 0;
            for (Integer integer : value)
            {
                attributes[row][col] = integer;
                col++;
            }
            row++;
        }

        int[] outputValues = new int[output.size()];
        for (int i = 0; i < output.size(); i++)
        {
            outputValues[i] = output.get(i);
        }

        bufferedReader.close();
        return new Input(attributes, outputValues, attributeNames);
    }

    /**
     *
     * @param output
     * @param kindOfData
     */
    private static void calculateAccuracyPrintResults(Output output, String kindOfData, int i)
    {

        System.out.println("--------------------"+kindOfData+"--------------------------");
        System.out.println("Number of "+kindOfData+" instances = " + output.getInput().getAttributesAndValues()[0].length);
        System.out.println("Number of  "+kindOfData+" attributes = " + output.getInput().getAttributesAndValues().length);
        if (kindOfData.equals("TRAINING DATA"))
        {
            System.out.println("Total number of nodes in the tree = " + getNodeCount(output.decisionTree));
            System.out.println("Number of leaf nodes in the tree = " + getLeafCount(output.decisionTree));
        }
        double accuracy = calculateAccuracy(output.getDecisionTree(), output.getInput());
        System.out.println("Accuracy of the model on the "+kindOfData+" set = " + accuracy);

        if (kindOfData.equals("TRAINING DATA"))
        {
            StringBuilder sb = new StringBuilder("");
            printTree(output.getDecisionTree(), sb);
            if (i == 1)
            System.out.println(sb.toString());
        }

        List<Tree> allLeafNodes = getAllLeafNodes(output.getDecisionTree());

        int totalNoOfLeaves = allLeafNodes.size();

        int sumOfDepthOfLeafNodes = 0;

        for (Tree allLeafNode : allLeafNodes)
        {
            sumOfDepthOfLeafNodes += allLeafNode.getParentCount();
        }

        System.out.println("Average Depth "+((double)totalNoOfLeaves/(double)sumOfDepthOfLeafNodes));


    }

    /**
     *
     * @param decisionTree
     * @param input
     * @return
     */
    private static double calculateAccuracy(Tree decisionTree, Input input)
    {

        int correctlyClassified = 0;
        for (int i = 0; i < input.getOutputValues().length; i++)
        {
            Map<String, Integer> inputSequence = new LinkedHashMap<>();
            for (int j = 0; j < input.getAttributesAndValues().length; j++)
            {
                inputSequence.put(input.getAttributeNames()[j], input.getAttributesAndValues()[j][i]);
            }

            Integer predictedOutput = evaluateInputSequence(inputSequence, decisionTree, null);
            if (predictedOutput == null)
                System.out.println("Something is very wrong");

            if (predictedOutput == input.getOutputValues()[i])
                ++correctlyClassified;
            //System.out.println("Input Sequence :: " + inputSequence + " Actual Value :: " + output.getOutputValues()[i] + " Predicted Value :: " + predictedOutput);
        }

        return (double) correctlyClassified/ (double) input.getAttributesAndValues()[0].length;
    }

    /**
     *
     * @param inputSequence
     * @param decisionTree
     * @param parentTree
     * @return
     */
    private static Integer evaluateInputSequence(Map<String, Integer> inputSequence, Tree decisionTree, Tree parentTree)
    {
        if (decisionTree == null)
        {
            return Integer.parseInt(parentTree.outputValue);
        }
        if (decisionTree.isLeaf())
        {
            return Integer.parseInt(decisionTree.outputValue);
        }

        Integer input = inputSequence.get(decisionTree.data);
        if (input == null)
            System.out.println("Something wrong");

        Integer output = evaluateInputSequence(inputSequence, decisionTree.getChild(input + ""), decisionTree.getParent());
        return output;
    }

    /**
     *
     * @param decisionTree
     * @return
     */
    private static List<Tree> getAllLeafNodes(Tree decisionTree)
    {
        if (decisionTree == null)
            return null;

        List<Tree> allLeaves = new ArrayList<>();
        for (Tree tree : decisionTree.getAllChildren())
        {
            if (tree.isLeaf())
                allLeaves.add(tree);
        }

        return allLeaves;
    }

    /**
     *
     * @param decisionTree
     * @return
     */
    private static int getLeafCount(Tree decisionTree)
    {
        int leafCount = 0;
        for (Tree tree : decisionTree.getAllChildren())
        {
            if (tree.isLeaf())
                leafCount++;
        }

        return leafCount;
    }

    /**
     *
     * @param decisionTree
     * @return
     */
    private static int getNodeCount(Tree decisionTree)
    {
        if (decisionTree == null)
            return 0;

        return decisionTree.getAllChildren().size();

    }

    /**
     *
     * @param decisionTree
     * @param sb
     */
    private static void printTree(Tree decisionTree, StringBuilder sb)
    {

        if (decisionTree.isLeaf())
        {
            sb.append(" "+ decisionTree.getOutputValue());
        }
        Map<String, Tree> children = decisionTree.getChildren();

        for (Map.Entry<String, Tree> stringTreeEntry : children.entrySet())
        {
            sb.append("\n");
            int noOfParents = decisionTree.getParentCount();
            for ( int i = 0; i < noOfParents; i++)
            {
                sb.append("|"+"\t");
            }
            sb.append(decisionTree.data + " =  ");
            String zeroOr1 = stringTreeEntry.getKey();
            Tree child = stringTreeEntry.getValue();

            sb.append(zeroOr1 + " : ");

            printTree(child, sb);
        }


    }


    /**
     *  @param attributesAndValues
     * @param outputValues
     * @param attributeNames
     * @param decisionTree
     * @param from
     * @param algorithm
     */
    static void constructDecisionTree(int[][] attributesAndValues, int[] outputValues, String[] attributeNames, Tree decisionTree, String from, String algorithm)
    {

        if (attributeNames == null || attributeNames.length == 0)
        {

            double[] noOf0sAnd1s = getNoOf0sAnd1s(outputValues);
            String s = "0:" + noOf0sAnd1s[0] + ":1:" + noOf0sAnd1s[1];

            Tree childTree = new Tree(s, from);
            String output;
            if (noOf0sAnd1s[0] > noOf0sAnd1s[1])
            {
                output = "0";
            }
            else
            {
                output = "1";
            }
            childTree.setOutputValue(output);
            childTree.setExtraDats(s);
            decisionTree.addChild(childTree, from);
            return;
        }

        double[] noOf0sAnd1s1 = getNoOf0sAnd1s(outputValues);
        if (noOf0sAnd1s1[0] == 0 || noOf0sAnd1s1[1] == 0 )
        {
            double[] noOf0sAnd1s = getNoOf0sAnd1s(outputValues);
            String s = " 0 : " + noOf0sAnd1s[0] + "  1 : " + noOf0sAnd1s[1];
            Tree childTree = new Tree(s, from);

            String output;
            if (noOf0sAnd1s[0] > noOf0sAnd1s[1])
            {
                output = "0";
            }
            else
            {
                output = "1";
            }
            childTree.setOutputValue(output);
            childTree.setExtraDats(s);
            decisionTree.addChild(childTree, from);
            return;
        }

        int indexWithMaxInfoGain = getIndexByAlgorithm(attributeNames, attributesAndValues, outputValues, algorithm);

        //if (!algorithm.equals("ID3"))
          //  System.out.println("INDEX WITH MAX " + indexWithMaxInfoGain);

        if (decisionTree.isDummy())
        {
            double[] noOf0sAnd1s = getNoOf0sAnd1s(outputValues);
            String output;
            if (noOf0sAnd1s[0] > noOf0sAnd1s[1])
            {
                output = "0";
            }
            else
            {
                output = "1";
            }
            decisionTree.setExtraDats("0s :" + noOf0sAnd1s[0] + "  1s :" + noOf0sAnd1s[1]);
            decisionTree.setData(attributeNames[indexWithMaxInfoGain], from);
            decisionTree.setOutputValue(output);
        }
        else
        {
            Tree childTree = new Tree(attributeNames[indexWithMaxInfoGain], from);
            double[] noOf0sAnd1s = getNoOf0sAnd1s(outputValues);
            String output;
            if (noOf0sAnd1s[0] > noOf0sAnd1s[1])
            {
                output = "0";
            }
            else
            {
                output = "1";
            }
            childTree.setOutputValue(output);
            childTree.setExtraDats("0s :" + noOf0sAnd1s[0] + "  1s :" + noOf0sAnd1s[1]);
            decisionTree.addChild(childTree, from);
            decisionTree = childTree;
        }




        int[] attributesValuesWithMaxInfoGain = attributesAndValues[indexWithMaxInfoGain];

        List<Integer> indicesWith0s = getIndicesWith(attributesValuesWithMaxInfoGain,0);
        List<Integer> indicesWith1s = getIndicesWith(attributesValuesWithMaxInfoGain,1);
        int[][] newAttributeValues0 = new int[attributesAndValues.length -1][indicesWith0s.size()];
        int[][] newAttributeValues1 = new int[attributesAndValues.length -1][indicesWith1s.size()];

        int[] outputValues0 = new int[indicesWith0s.size()];
        int[] outputValues1 = new int[indicesWith1s.size()];

        String[] newAttributeNames = new String[attributeNames.length - 1];


        int row = 0;
        for (int i = 0; i < attributesAndValues.length; i++)
        {

            int j0 = 0;
            int j1 = 0;
            if (i == indexWithMaxInfoGain)
            {
                continue;
            }
            newAttributeNames[row] = attributeNames[i];


            for (Integer indicesWith0 : indicesWith0s)
            {

                newAttributeValues0[row][j0] = attributesAndValues[i][indicesWith0];
                outputValues0[j0] = outputValues[indicesWith0];
                j0++;
            }

            for (Integer indicesWith1 : indicesWith1s)
            {
                newAttributeValues1[row][j1] = attributesAndValues[i][indicesWith1];
                outputValues1[j1] = outputValues[indicesWith1];
                j1++;
            }

            row++;

        }


        if (newAttributeNames.length == 0)
        {
            for (int i = 0; i < indicesWith0s.size(); i++)
            {
                outputValues0[i] = outputValues[indicesWith0s.get(i)];
            }

            for (int i = 0; i < indicesWith1s.size(); i++)
            {
                outputValues1[i] = outputValues[indicesWith1s.get(i)];
            }
        }




        constructDecisionTree(newAttributeValues0, outputValues0, newAttributeNames, decisionTree, "0", algorithm);
        constructDecisionTree(newAttributeValues1, outputValues1, newAttributeNames, decisionTree, "1", algorithm);

    }

    private static int getIndexByAlgorithm(String[] attributeNames, int[][] attributesAndValues, int[] outputValues, String algorithm)
    {
        return algorithm.equals("ID3") ? getIndexWithMaxInfoGain(attributeNames, attributesAndValues, outputValues) : getRandomAttributeIndex(attributeNames);
    }

    private static int getRandomAttributeIndex(String[] attributeNames)
    {
        if (attributeNames == null || attributeNames.length == 0)
            return -1;

        //System.out.println("GETTING RANDOM");
        return random.nextInt(attributeNames.length);
    }

    /**
     *
     * @param attributesValuesWithMaxInfoGain
     * @param integer
     * @return
     */
    private static List<Integer> getIndicesWith(int[] attributesValuesWithMaxInfoGain, int integer)
    {
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < attributesValuesWithMaxInfoGain.length ; i++)
        {
            if (attributesValuesWithMaxInfoGain[i] == integer)
                indices.add(i);
        }

        return indices;
    }


    /**
     *
     * @param attributeNames
     * @param attributesAndValues
     * @param outputValues
     * @return
     */
    static int getIndexWithMaxInfoGain(String[] attributeNames, int[][] attributesAndValues, int[] outputValues)
    {
        int indexWithMaxInfoGain = -1;
        double maxInfoGain = Integer.MIN_VALUE;
        for (int i = 0; i < attributesAndValues.length; i++)
        {

            double infoGain = computeInformationGain(attributesAndValues[i], outputValues);

//            if (infoGain <= 0)
//                continue;

            if (infoGain > maxInfoGain)
            {
                maxInfoGain = infoGain;
                indexWithMaxInfoGain = i;
            }
        }

        return indexWithMaxInfoGain;
    }

    /**
     *
     * @param attributeValues
     * @param outputValues
     * @return
     */
    static double computeInformationGain(int[] attributeValues, int[] outputValues)
    {
        double entropyOfSystem = computeEntropy(computeProbability(outputValues));

        double avgEntropyofChild = computeAvgEntropyOfChild(attributeValues, outputValues);

        return (entropyOfSystem - avgEntropyofChild);
    }

    /**
     *
     * @param attributeValues
     * @param outputValues
     * @return
     */
    static double computeAvgEntropyOfChild(int[] attributeValues, int[] outputValues)
    {

        double[] conditionalProbabilityWhenXis0 = computeConditionalProbability(attributeValues, outputValues, 0);

        double entropyWhenXIs0 = computeEntropy(conditionalProbabilityWhenXis0);

        double[] conditionalProbabilityWhenXis1 = computeConditionalProbability(attributeValues, outputValues, 1);

        double entropyWhenXIs1 = computeEntropy(conditionalProbabilityWhenXis1);

        double[] noOf0sAnd1s = getNoOf0sAnd1s(attributeValues);



        return ((noOf0sAnd1s[0]/attributeValues.length) * entropyWhenXIs0) + ((noOf0sAnd1s[1]/attributeValues.length) * entropyWhenXIs1);

    }

    /**
     *
     * @param attributeValues
     * @param outputValues
     * @param occurringEvent
     * @return
     */
    static double[] computeConditionalProbability(int[] attributeValues, int[] outputValues, int occurringEvent)
    {

        int totalNoOfOccurringEvent = 0;
        List<Integer> indicesOfOccurringEvents = new ArrayList<>();
        for (int i = 0; i < attributeValues.length; i++)
        {
            if (attributeValues[i] == occurringEvent)
            {
                totalNoOfOccurringEvent++;
                indicesOfOccurringEvents.add(i);
            }
        }

        int[] outputsWhenThisEventOccurs = new int[indicesOfOccurringEvents.size()];

        for (int i = 0; i < outputsWhenThisEventOccurs.length ; i++)
        {
            outputsWhenThisEventOccurs[i] = outputValues[indicesOfOccurringEvents.get(i)];
        }

        double[] noOf0sAnd1s = getNoOf0sAnd1s(outputsWhenThisEventOccurs);


        if (indicesOfOccurringEvents.isEmpty())
            return new double[]{0,0};

        return new double[]{noOf0sAnd1s[0]/totalNoOfOccurringEvent, noOf0sAnd1s[1]/totalNoOfOccurringEvent};
    }


    private static class Tree
    {
        String from;
        String data;
        Map<String,Tree> children = new HashMap<>();
        String outputValue;
        String extraDats;

        Tree parent = null;

        public Tree(String attributeName, String from)
        {
            this.data = attributeName;
            this.from = from;
        }

        public Tree()
        {

        }

        public Tree(Tree decisionTree)
        {
            this.from = decisionTree.getFrom();
            this.data = decisionTree.getData();
            this.outputValue = decisionTree.getOutputValue();
        }

        public String getData()
        {
            return data;
        }

        public String getFrom()
        {
            return from;
        }

        public Tree getParent()
        {
            return parent;
        }



        public void addChild(Tree childTree, String from)
        {
            childTree.setParent(this);
            this.children.put(from,childTree);
        }

        public boolean isDummy()
        {
            return data == null;
        }

        public void setData(String attributeName, String from)
        {
            this.data = attributeName;
            this.from = from;
        }

        public void setParent(Tree parent)
        {
            this.parent = parent;
        }

        public void setOutputValue(String outputValue)
        {
            this.outputValue = outputValue;
        }

        public Map<String, Tree> getChildren()
        {
            return children;
        }

        public String getOutputValue()
        {
            return outputValue;
        }

        public boolean isLeaf()
        {

            return children == null || children.isEmpty();
        }

        public int getParentCount()
        {
            int parentCount = 0;
            Tree par = parent;

            if (par == null)
                return 0;

            else
            {
                while (par != null)
                {
                    parentCount++;
                    par = par.parent;
                }
            }

            return parentCount;

        }

        public List<Tree> getAllChildren()
        {
            List<Tree> allChildren = new ArrayList<>();
            populateAllChildren(this, allChildren);
            return allChildren;
        }


        public void populateAllChildren(Tree tree, List<Tree> allChildren)
        {
            allChildren.add(tree);


            for (Tree child : tree.getChildren().values())
            {
                populateAllChildren(child, allChildren);
            }
        }

        public Tree getChild(String s)
        {
            return children.get(s);
        }

        public void removeChild(String from)
        {
            children.remove(from);
        }

        @Override
        public boolean equals(Object o)
        {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;

            Tree tree = (Tree) o;

            if (from != null ? !from.equals(tree.from) : tree.from != null) return false;
            if (data != null ? !data.equals(tree.data) : tree.data != null) return false;
            //  if (children != null ? !children.equals(tree.children) : tree.children != null) return false;
            if (outputValue != null ? !outputValue.equals(tree.outputValue) : tree.outputValue != null) return false;
            return parent != null ? parent.equals(tree.parent) : tree.parent == null;

        }

        @Override
        public int hashCode()
        {
            int result = from != null ? from.hashCode() : 0;
            result = 31 * result + (data != null ? data.hashCode() : 0);
//            result = 31 * result + (children != null ? children.hashCode() : 0);
            result = 31 * result + (outputValue != null ? outputValue.hashCode() : 0);
            result = 31 * result + (parent != null ? parent.hashCode() : 0);
            return result;
        }

        public void removeChildren()
        {
            this.children = new HashMap<>();
        }

        public void setChildren(Map<String, Tree> children)
        {
            this.children = children;
        }

        public void setExtraDats(String extraDats)
        {
            this.extraDats = extraDats;
        }
    }

    public static class Input
    {
        private int[][] attributesAndValues;
        private int[] outputValues;
        private String[] attributeNames;

        public Input(int[][] attributesAndValues, int[] outputValues, String[] attributeNames)
        {
            this.attributesAndValues = attributesAndValues;
            this.outputValues = outputValues;
            this.attributeNames = attributeNames;
        }

        public int[] getOutputValues()
        {
            return outputValues;
        }

        public int[][] getAttributesAndValues()
        {
            return attributesAndValues;
        }

        public String[] getAttributeNames()
        {
            return attributeNames;
        }
    }


    public static class Output
    {
        private Input input;
        private Tree decisionTree;

        public Output(Input input, Tree decisionTree)
        {
            this.input = input;
            this.decisionTree = decisionTree;
        }

        public Input getInput()
        {
            return input;
        }

        public Tree getDecisionTree()
        {
            return decisionTree;
        }

        public int[] getOutputValues()
        {
            return input.getOutputValues();
        }

        public int[][] getAttributes()
        {
            return input.getAttributesAndValues();
        }

        public String[] getAttributeNames()
        {
            return input.getAttributeNames();
        }
    }


    private static class TrainingAndValidationInput
    {
        Input trainingInput;

        Input validationInput;

        public TrainingAndValidationInput(Input trainingInput, Input validationInput)
        {
            this.trainingInput = trainingInput;
            this.validationInput = validationInput;
        }

        public Input getTrainingInput()
        {
            return trainingInput;
        }

        public Input getValidationInput()
        {
            return validationInput;
        }
    }



}

