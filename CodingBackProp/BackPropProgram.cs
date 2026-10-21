using System;

namespace CodingBackProp
{
    class BackPropProgram
    {
        static void Main(string[] args)
        {
            Console.WriteLine("\nBegin neural network back-propagation demo");

            int numInput = 4;   // number of features
            int numHidden = 5;
            int numOutput = 3;  // number of classes
            int numRows = 1000;
            int seed = 1;

            Console.WriteLine("\nGenerating " + numRows + " artificial data items with " + numInput + " features");
            double[][] allData = MakeAllData(numInput, numHidden, numOutput, numRows, seed);
            Console.WriteLine("Done");

            Console.WriteLine("\nCreating train (80%) and test (20%) matrices");
            double[][] trainData;
            double[][] testData;
            SplitTrainTest(allData, 0.80, seed, out trainData, out testData);
            Console.WriteLine("Done\n");

            Console.WriteLine("Training data sample:");
            ShowMatrix(trainData, 4, 2, true);

            Console.WriteLine("Creating a " + numInput + "-" + numHidden + "-" + numOutput + " neural network");
            NeuralNetwork nn = new NeuralNetwork(numInput, numHidden, numOutput);

            int maxEpochs = 1000;
            double learnRate = 0.01;
            double momentum = 0.01;

            // [MODIFICATION 2: Weight Decay Parameter]
            // Regularization penalty: shrinks weights towards zero to reduce overfitting
            double weightDecay = 0.0001;

            Console.WriteLine("\nSetting maxEpochs   = " + maxEpochs);
            Console.WriteLine("Setting learnRate   = " + learnRate.ToString("F4"));
            Console.WriteLine("Setting momentum    = " + momentum.ToString("F4"));
            Console.WriteLine("Setting weightDecay = " + weightDecay.ToString("F4"));

            Console.WriteLine("\nStarting training");
            double[] weights = nn.Train(trainData, maxEpochs, learnRate, momentum, weightDecay);
            Console.WriteLine("Done");

            Console.WriteLine("\nFinal neural network model weights and biases:\n");
            ShowVector(weights, 2, 10, true);

            double trainAcc = nn.Accuracy(trainData);
            Console.WriteLine("\nFinal accuracy on training data = " + trainAcc.ToString("F4"));

            double testAcc = nn.Accuracy(testData);
            Console.WriteLine("Final accuracy on test data     = " + testAcc.ToString("F4"));

            Console.WriteLine("\nEnd back-propagation demo\n");
            Console.ReadLine();
        }

        public static void ShowMatrix(double[][] matrix, int numRows, int decimals, bool indices)
        {
            int len = matrix.Length.ToString().Length;
            for (int i = 0; i < numRows; ++i)
            {
                if (indices) Console.Write("[" + i.ToString().PadLeft(len) + "]  ");
                for (int j = 0; j < matrix[i].Length; ++j)
                {
                    double v = matrix[i][j];
                    if (v >= 0.0) Console.Write(" ");
                    Console.Write(v.ToString("F" + decimals) + "  ");
                }
                Console.WriteLine("");
            }

            if (numRows < matrix.Length)
            {
                Console.WriteLine(". . .");
                int lastRow = matrix.Length - 1;
                if (indices) Console.Write("[" + lastRow.ToString().PadLeft(len) + "]  ");
                for (int j = 0; j < matrix[lastRow].Length; ++j)
                {
                    double v = matrix[lastRow][j];
                    if (v >= 0.0) Console.Write(" ");
                    Console.Write(v.ToString("F" + decimals) + "  ");
                }
            }
            Console.WriteLine("\n");
        }

        public static void ShowVector(double[] vector, int decimals, int lineLen, bool newLine)
        {
            for (int i = 0; i < vector.Length; ++i)
            {
                if (i > 0 && i % lineLen == 0) Console.WriteLine("");
                if (vector[i] >= 0) Console.Write(" ");
                Console.Write(vector[i].ToString("F" + decimals) + " ");
            }
            if (newLine) Console.WriteLine("");
        }

        static double[][] MakeAllData(int numInput, int numHidden, int numOutput, int numRows, int seed)
        {
            Random rnd = new Random(seed);
            int numWeights = (numInput * numHidden) + numHidden + (numHidden * numOutput) + numOutput;
            double[] weights = new double[numWeights];
            for (int i = 0; i < numWeights; ++i)
                weights[i] = 20.0 * rnd.NextDouble() - 10.0;

            double[][] result = new double[numRows][];
            for (int i = 0; i < numRows; ++i)
                result[i] = new double[numInput + numOutput];

            NeuralNetwork gnn = new NeuralNetwork(numInput, numHidden, numOutput);
            gnn.SetWeights(weights);

            for (int r = 0; r < numRows; ++r)
            {
                double[] inputs = new double[numInput];
                for (int i = 0; i < numInput; ++i)
                    inputs[i] = 20.0 * rnd.NextDouble() - 10.0;

                double[] outputs = gnn.ComputeOutputs(inputs);
                double[] oneOfN = new double[numOutput];

                int maxIndex = 0;
                double maxValue = outputs[0];
                for (int i = 0; i < numOutput; ++i)
                {
                    if (outputs[i] > maxValue)
                    {
                        maxIndex = i;
                        maxValue = outputs[i];
                    }
                }
                oneOfN[maxIndex] = 1.0;

                int c = 0;
                for (int i = 0; i < numInput; ++i)
                    result[r][c++] = inputs[i];
                for (int i = 0; i < numOutput; ++i)
                    result[r][c++] = oneOfN[i];
            }
            return result;
        }

        static void SplitTrainTest(double[][] allData, double trainPct, int seed, out double[][] trainData, out double[][] testData)
        {
            Random rnd = new Random(seed);
            int totRows = allData.Length;
            int numTrainRows = (int)(totRows * trainPct);
            int numTestRows = totRows - numTrainRows;
            trainData = new double[numTrainRows][];
            testData = new double[numTestRows][];

            double[][] copy = new double[allData.Length][];
            for (int i = 0; i < copy.Length; ++i)
                copy[i] = allData[i];

            for (int i = 0; i < copy.Length; ++i)
            {
                int r = rnd.Next(i, copy.Length);
                double[] tmp = copy[r];
                copy[r] = copy[i];
                copy[i] = tmp;
            }
            for (int i = 0; i < numTrainRows; ++i)
                trainData[i] = copy[i];

            for (int i = 0; i < numTestRows; ++i)
                testData[i] = copy[i + numTrainRows];
        }
    }

    public class NeuralNetwork
    {
        private int numInput;
        private int numHidden;
        private int numOutput;

        private double[] inputs;
        private double[][] ihWeights;
        private double[] hBiases;
        private double[] hOutputs;

        private double[][] hoWeights;
        private double[] oBiases;
        private double[] outputs;

        private Random rnd;

        public NeuralNetwork(int numInput, int numHidden, int numOutput)
        {
            this.numInput = numInput;
            this.numHidden = numHidden;
            this.numOutput = numOutput;

            this.inputs = new double[numInput];
            this.ihWeights = MakeMatrix(numInput, numHidden, 0.0);
            this.hBiases = new double[numHidden];
            this.hOutputs = new double[numHidden];

            this.hoWeights = MakeMatrix(numHidden, numOutput, 0.0);
            this.oBiases = new double[numOutput];
            this.outputs = new double[numOutput];

            this.rnd = new Random(0);
            this.InitializeWeights();
        }

        private static double[][] MakeMatrix(int rows, int cols, double v)
        {
            double[][] result = new double[rows][];
            for (int r = 0; r < result.Length; ++r)
                result[r] = new double[cols];
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j)
                    result[i][j] = v;
            return result;
        }

        private void InitializeWeights()
        {
            int numWeights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;
            double[] initialWeights = new double[numWeights];
            for (int i = 0; i < initialWeights.Length; ++i)
                initialWeights[i] = (0.001 - 0.0001) * rnd.NextDouble() + 0.0001;
            this.SetWeights(initialWeights);
        }

        public void SetWeights(double[] weights)
        {
            int numWeights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;
            if (weights.Length != numWeights)
                throw new Exception("Bad weights array in SetWeights");

            int k = 0;
            for (int i = 0; i < numInput; ++i)
                for (int j = 0; j < numHidden; ++j)
                    ihWeights[i][j] = weights[k++];
            for (int i = 0; i < numHidden; ++i)
                hBiases[i] = weights[k++];
            for (int i = 0; i < numHidden; ++i)
                for (int j = 0; j < numOutput; ++j)
                    hoWeights[i][j] = weights[k++];
            for (int i = 0; i < numOutput; ++i)
                oBiases[i] = weights[k++];
        }

        public double[] GetWeights()
        {
            int numWeights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;
            double[] result = new double[numWeights];
            int k = 0;
            for (int i = 0; i < ihWeights.Length; ++i)
                for (int j = 0; j < ihWeights[0].Length; ++j)
                    result[k++] = ihWeights[i][j];
            for (int i = 0; i < hBiases.Length; ++i)
                result[k++] = hBiases[i];
            for (int i = 0; i < hoWeights.Length; ++i)
                for (int j = 0; j < hoWeights[0].Length; ++j)
                    result[k++] = hoWeights[i][j];
            for (int i = 0; i < oBiases.Length; ++i)
                result[k++] = oBiases[i];
            return result;
        }

        public double[] ComputeOutputs(double[] xValues)
        {
            double[] hSums = new double[numHidden];
            double[] oSums = new double[numOutput];

            for (int i = 0; i < xValues.Length; ++i)
                this.inputs[i] = xValues[i];

            for (int j = 0; j < numHidden; ++j)
                for (int i = 0; i < numInput; ++i)
                    hSums[j] += this.inputs[i] * this.ihWeights[i][j];

            for (int i = 0; i < numHidden; ++i)
                hSums[i] += this.hBiases[i];

            for (int i = 0; i < numHidden; ++i)
                this.hOutputs[i] = HyperTan(hSums[i]);

            for (int j = 0; j < numOutput; ++j)
                for (int i = 0; i < numHidden; ++i)
                    oSums[j] += hOutputs[i] * hoWeights[i][j];

            for (int i = 0; i < numOutput; ++i)
                oSums[i] += oBiases[i];

            double[] softOut = Softmax(oSums);
            Array.Copy(softOut, outputs, softOut.Length);

            double[] retResult = new double[numOutput];
            Array.Copy(this.outputs, retResult, retResult.Length);
            return retResult;
        }

        private static double HyperTan(double x)
        {
            if (x < -20.0) return -1.0;
            else if (x > 20.0) return 1.0;
            else return Math.Tanh(x);
        }

        private static double[] Softmax(double[] oSums)
        {
            // Determine max value for numerical stability (prevents Math.Exp overflow)
            double max = oSums[0];
            for (int i = 1; i < oSums.Length; ++i)
                if (oSums[i] > max) max = oSums[i];

            double sum = 0.0;
            for (int i = 0; i < oSums.Length; ++i)
                sum += Math.Exp(oSums[i] - max);

            double[] result = new double[oSums.Length];
            for (int i = 0; i < oSums.Length; ++i)
                result[i] = Math.Exp(oSums[i] - max) / sum;

            return result;
        }

        public double[] Train(double[][] trainData, int maxEpochs, double learnRate, double momentum, double weightDecay)
        {
            double[][] hoGrads = MakeMatrix(numHidden, numOutput, 0.0);
            double[] obGrads = new double[numOutput];

            double[][] ihGrads = MakeMatrix(numInput, numHidden, 0.0);
            double[] hbGrads = new double[numHidden];

            double[] oSignals = new double[numOutput];
            double[] hSignals = new double[numHidden];

            double[][] ihPrevWeightsDelta = MakeMatrix(numInput, numHidden, 0.0);
            double[] hPrevBiasesDelta = new double[numHidden];
            double[][] hoPrevWeightsDelta = MakeMatrix(numHidden, numOutput, 0.0);
            double[] oPrevBiasesDelta = new double[numOutput];

            int epoch = 0;
            double[] xValues = new double[numInput];
            double[] tValues = new double[numOutput];

            int[] sequence = new int[trainData.Length];
            for (int i = 0; i < sequence.Length; ++i)
                sequence[i] = i;

            int errInterval = maxEpochs / 10;
            while (epoch < maxEpochs)
            {
                ++epoch;

                if (epoch % errInterval == 0 && epoch < maxEpochs)
                {
                    double trainErr = CrossEntropyError(trainData);
                    Console.WriteLine("epoch = " + epoch + "  cross-entropy error = " + trainErr.ToString("F4"));
                }

                Shuffle(sequence);
                for (int ii = 0; ii < trainData.Length; ++ii)
                {
                    int idx = sequence[ii];
                    Array.Copy(trainData[idx], xValues, numInput);
                    Array.Copy(trainData[idx], numInput, tValues, 0, numOutput);
                    ComputeOutputs(xValues);

                    // =========================================================================
                    // [MODIFICATION 1: Cross-Entropy with Softmax Output Signal]
                    // The derivative of cross-entropy loss with softmax activation simplifies to:
                    //   dL/dz = output - target  (or target - output for gradient ascent)
                    // No separate derivative multiplication (output * (1 - output)) is needed!
                    // =========================================================================
                    for (int k = 0; k < numOutput; ++k)
                    {
                        oSignals[k] = tValues[k] - outputs[k];
                    }

                    // Hidden-to-output weight gradients
                    for (int j = 0; j < numHidden; ++j)
                        for (int k = 0; k < numOutput; ++k)
                            hoGrads[j][k] = oSignals[k] * hOutputs[j];

                    // Output bias gradients
                    for (int k = 0; k < numOutput; ++k)
                        obGrads[k] = oSignals[k] * 1.0;

                    // Hidden node signals
                    for (int j = 0; j < numHidden; ++j)
                    {
                        double derivative = (1 + hOutputs[j]) * (1 - hOutputs[j]); // tanh derivative
                        double sum = 0.0;
                        for (int k = 0; k < numOutput; ++k)
                            sum += oSignals[k] * hoWeights[j][k];

                        hSignals[j] = derivative * sum;
                    }

                    // Input-to-hidden weight gradients
                    for (int i = 0; i < numInput; ++i)
                        for (int j = 0; j < numHidden; ++j)
                            ihGrads[i][j] = hSignals[j] * inputs[i];

                    // Hidden bias gradients
                    for (int j = 0; j < numHidden; ++j)
                        hbGrads[j] = hSignals[j] * 1.0;

                    // =========================================================================
                    // [MODIFICATION 2: Weight Decay Updates]
                    // Subtract (weightDecay * weight) to shrink weights (L2 regularization).
                    // Biases are NOT regularized (standard deep learning practice).
                    // =========================================================================

                    // Update input-to-hidden weights with weight decay
                    for (int i = 0; i < numInput; ++i)
                    {
                        for (int j = 0; j < numHidden; ++j)
                        {
                            double delta = (ihGrads[i][j] - (weightDecay * ihWeights[i][j])) * learnRate;
                            ihWeights[i][j] += delta;
                            ihWeights[i][j] += ihPrevWeightsDelta[i][j] * momentum;
                            ihPrevWeightsDelta[i][j] = delta;
                        }
                    }

                    // Update hidden biases (no weight decay on biases)
                    for (int j = 0; j < numHidden; ++j)
                    {
                        double delta = hbGrads[j] * learnRate;
                        hBiases[j] += delta;
                        hBiases[j] += hPrevBiasesDelta[j] * momentum;
                        hPrevBiasesDelta[j] = delta;
                    }

                    // Update hidden-to-output weights with weight decay
                    for (int j = 0; j < numHidden; ++j)
                    {
                        for (int k = 0; k < numOutput; ++k)
                        {
                            double delta = (hoGrads[j][k] - (weightDecay * hoWeights[j][k])) * learnRate;
                            hoWeights[j][k] += delta;
                            hoWeights[j][k] += hoPrevWeightsDelta[j][k] * momentum;
                            hoPrevWeightsDelta[j][k] = delta;
                        }
                    }

                    // Update output biases (no weight decay on biases)
                    for (int k = 0; k < numOutput; ++k)
                    {
                        double delta = obGrads[k] * learnRate;
                        oBiases[k] += delta;
                        oBiases[k] += oPrevBiasesDelta[k] * momentum;
                        oPrevBiasesDelta[k] = delta;
                    }

                } // each training item

            } // epoch loop

            return GetWeights();
        }

        private void Shuffle(int[] sequence)
        {
            for (int i = 0; i < sequence.Length; ++i)
            {
                int r = this.rnd.Next(i, sequence.Length);
                int tmp = sequence[r];
                sequence[r] = sequence[i];
                sequence[i] = tmp;
            }
        }

        // [MODIFICATION 1: Cross-Entropy Error Function]
        // Replaces Mean Squared Error: Loss = -Sum( target * ln(output) )
        public double CrossEntropyError(double[][] trainData)
        {
            double sumError = 0.0;
            double[] xValues = new double[numInput];
            double[] tValues = new double[numOutput];

            for (int i = 0; i < trainData.Length; ++i)
            {
                Array.Copy(trainData[i], xValues, numInput);
                Array.Copy(trainData[i], numInput, tValues, 0, numOutput);
                double[] yValues = this.ComputeOutputs(xValues);

                for (int j = 0; j < numOutput; ++j)
                {
                    // + 1.0e-15 prevents Math.Log(0) which returns -Infinity
                    sumError += -tValues[j] * Math.Log(yValues[j] + 1.0e-15);
                }
            }
            return sumError / trainData.Length;
        }

        public double Accuracy(double[][] testData)
        {
            int numCorrect = 0;
            int numWrong = 0;
            double[] xValues = new double[numInput];
            double[] tValues = new double[numOutput];

            for (int i = 0; i < testData.Length; ++i)
            {
                Array.Copy(testData[i], xValues, numInput);
                Array.Copy(testData[i], numInput, tValues, 0, numOutput);
                double[] yValues = this.ComputeOutputs(xValues);

                int maxIndex = MaxIndex(yValues);
                int tMaxIndex = MaxIndex(tValues);

                if (maxIndex == tMaxIndex)
                    ++numCorrect;
                else
                    ++numWrong;
            }
            return (numCorrect * 1.0) / (numCorrect + numWrong);
        }

        private static int MaxIndex(double[] vector)
        {
            int bigIndex = 0;
            double biggestVal = vector[0];
            for (int i = 0; i < vector.Length; ++i)
            {
                if (vector[i] > biggestVal)
                {
                    biggestVal = vector[i];
                    bigIndex = i;
                }
            }
            return bigIndex;
        }
    }
}