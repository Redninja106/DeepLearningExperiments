using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepLearningExperiments;
internal abstract class ActivationFunction
{
    public abstract float[] Evaluate(float[] x, float[]? destination = null);
    public abstract float[] EvaluateGradient(float[] x, float[]? destination = null);
}

class ReLU : ActivationFunction
{
    public override float[] Evaluate(float[] x, float[]? destination = null)
    {
        destination ??= new float[x.Length];

        for (int i = 0; i < x.Length; i++)
        {
            destination[i] = float.Max(0, x[i]);
        }
        return destination;
    }

    public override float[] EvaluateGradient(float[] x, float[]? destination = null)
    {
        destination ??= new float[x.Length];

        for (int i = 0; i < x.Length; i++)
        {
            destination[i] = x[i] > 0 ? 1 : 0;
        }
        return destination;
    }
}

class Identity : ActivationFunction
{
    public override float[] Evaluate(float[] x, float[]? destination = null)
    {
        destination ??= new float[x.Length];
        Array.Copy(x, destination, x.Length);
        return destination;
    }

    public override float[] EvaluateGradient(float[] x, float[]? destination = null)
    {
        destination ??= new float[x.Length];
        Array.Fill(destination, 1f);
        return destination;
    }
}
