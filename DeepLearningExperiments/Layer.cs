using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepLearningExperiments;
internal abstract class Layer
{
    /// <summary>
    /// Also returns output size.
    /// </summary>
    public virtual int SetInputSize(int inputSize)
    {
        return inputSize;
    }

    public virtual void RandomInitialize(Random random)
    {
    }

    public abstract float[] Evaluate(float[] input);
}

class InputLayer : Layer
{
    private int size;

    public InputLayer(int size)
    {
        this.size = size;
    }

    public override int SetInputSize(int inputSize)
    {
        return size;
    }

    public override float[] Evaluate(float[] input)
    {
        if (input.Length != size)
        {
            throw new Exception($"Expected input size {size}, but got input size {input}!");
        }

        return input;
    }
}

class DeepLayer : Layer
{
    public float[] z;
    public float[] a;
    public float[,] weights;
    public float[] biases;
    int size;
    public ActivationFunction ActivationFunction;

    public DeepLayer(int size, ActivationFunction activationFunction)
    {
        this.ActivationFunction = activationFunction;
        this.size = size;
        z = new float[size];
        a = new float[size];
        biases = new float[size];
    }

    public override void RandomInitialize(Random random)
    {
        int fan_in = weights.GetLength(1);
        int fan_out = weights.GetLength(0);
        float limit = MathF.Sqrt(6f / (fan_in + fan_out));

        for (int i = 0; i < weights.GetLength(0); i++)
        {
            for (int j = 0; j < weights.GetLength(1); j++)
            {
                weights[i, j] = (random.NextSingle() * 2 - 1f) * limit;
            }
        }

        for (int i = 0; i < biases.GetLength(0); i++)
        {
            biases[i] = 0;
        }
    }

    public override int SetInputSize(int inputSize)
    {
        weights = new float[inputSize, size];
        return size;
    }

    public override float[] Evaluate(float[] previousLayer)
    {
        Matrix.Multiply(
            weights,
            previousLayer, 
            destination: z
            );

        Vector.Add(
            z,
            biases,
            destination: z
            );

        ActivationFunction.Evaluate(z, a);
        return a;
    }
}

//class SigmoidLayer : Layer
//{
//    public override float[] Evaluate(float[] input)
//    {
//        for (int i = 0; i <= input.Length; i++) 
//        {
//            input[i] = 1f / (1f + float.Exp(input[i]));
//        }
//        return input;
//    }
//}

class ReLULayer : Layer
{
    float[] values;

    public override int SetInputSize(int inputSize)
    {
        values = new float[inputSize];
        return base.SetInputSize(inputSize);
    }

    public override float[] Evaluate(float[] input)
    {
        for (int i = 0; i <= input.Length; i++)
        {
            values[i] = float.Max(0f, input[i]);
        }
        return input;
    }
}

class SoftmaxLayer : Layer
{
    public override float[] Evaluate(float[] input)
    {
        float sum = 0f;
        for (int i = 0; i < input.Length; i++)
        {
            input[i] = float.Exp(input[i]);
            sum += input[i];
        }
        float f = 1f / sum;
        for (int i = 0; i < input.Length; i++)
        {
            input[i] *= f;
        }
        return input;
    }
}
