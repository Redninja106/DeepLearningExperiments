using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;
using System.Xml;

namespace DeepLearningExperiments;
internal class Model
{
    private Layer[] Layers;
    
    public Model(params Layer[] layers)
    {
        this.Layers = layers;

        int prevSize = 0;
        foreach (var layer in layers)
        {
            prevSize = layer.SetInputSize(prevSize);
            layer.RandomInitialize(Random.Shared);
        }
    }

    public float[] Evaluate(float[] input)
    {
        float[] current = input;
        foreach (var layer in Layers)
        {
            current = layer.Evaluate(current);
        }
        return current;
    }

    public void Train(float[][] x, float[][] y, int steps, int samplesPerStep = 10, float learningRate = 0.01f)
    {
        for (int i = 0; i < steps; i++)
        {
            int index = Random.Shared.Next(x.Length);

            float[] result = Evaluate(x[index]);
            
            if (result.Any(r => float.IsNaN(r) || float.IsInfinity(r)))
            {
                Console.WriteLine($"NaN/Inf detected in output at step {i}");
                return;
            }
            
            float loss = Loss(result, y[index]);
            
            if (i % 100 == 0)
                Console.WriteLine($"Step {i}: {loss}");
            
            float[] delta = Vector.Subtract(result, y[index]);

            for (int j = Layers.Length - 2; j > 0; j--)
            {
                if (Layers[j] is not DeepLayer layer)
                    continue;

                float[] activations;
                if (Layers[j - 1] is DeepLayer deepLayer)
                {
                    activations = deepLayer.a;
                }
                else if (Layers[j - 1] is InputLayer)
                {
                    activations = x[index];
                }
                else
                {
                    throw new Exception($"Unknown layer type at index {j - 1}");
                }

                float[,] dw = Matrix.Multiply(activations, delta);
                
                for (int ii = 0; ii < dw.GetLength(0); ii++)
                {
                    for (int jj = 0; jj < dw.GetLength(1); jj++)
                    {
                        dw[ii, jj] = Math.Clamp(dw[ii, jj], -5f, 5f);
                    }
                }
                
                Matrix.Multiply(dw, learningRate, destination: dw);
                Matrix.Subtract(layer.weights, dw, destination: layer.weights);
                
                float[] biasGrad = Vector.Multiply(delta, learningRate);
                for (int k = 0; k < biasGrad.Length; k++)
                {
                    biasGrad[k] = Math.Clamp(biasGrad[k], -5f * learningRate, 5f * learningRate);
                }
                Vector.Subtract(layer.biases, biasGrad, destination: layer.biases);

                if (j > 1)
                {
                    var wt = Matrix.Transpose(layer.weights);
                    float[] prevDelta = Matrix.Multiply(wt, delta);
                    
                    if (Layers[j - 1] is DeepLayer prevDeepLayer)
                    {
                        delta = Vector.Multiply(
                            prevDelta,
                            prevDeepLayer.ActivationFunction.EvaluateGradient(prevDeepLayer.z)
                        );
                    }
                    else
                    {
                        delta = prevDelta;
                    }
                }
            }
        }
    }

    public float Loss(float[] actual, float[] expected)
    {
        float sum = 0;
        for (int i = 0; i < actual.Length; i++)
        {
            sum += expected[i] * MathF.Log(MathF.Max(actual[i], 1e-7f));
        }
        return -sum;
    }
}
