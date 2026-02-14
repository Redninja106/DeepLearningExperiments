using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Text.Json.Serialization;
using System.Threading.Tasks;

namespace DeepLearningExperiments;

static class ArrayExtensions
{
    public static int Width(this float[,] array) => array.GetLength(0);
    public static int Height(this float[,] array) => array.GetLength(1);
}

static class Matrix
{
    public static float[,] Multiply(float[,] a, float[,] b, float[,]? destination = null)
    {
        if (a.Width() != b.Height())
        {
            throw new Exception($"Incorrect matrix size: width of a ({a.Width()}) must match height of b ({b.Height()})");
        }

        destination ??= new float[b.Width(), a.Height()];
        for (int i = 0; i < destination.Height(); i++)
        {
            for (int j = 0; j < destination.Width(); j++)
            {
                for (int k = 0; k < a.Width(); k++)
                {
                    destination[i, j] += a[k, j] * b[i, k];
                }
            }
        }

        return destination;
    }

    public static float[] Multiply(float[] v, float[,] b, float[]? destination = null)
    {
        // v (length = b.Height) * b (b.Width x b.Height)  -> output length = b.Width
        if (v.Length != b.Height())
        {
            throw new Exception($"Incorrect vector/matrix size: length of v ({v.Length}) must match height of b ({b.Height()})");
        }

        destination ??= new float[b.Width()];
        for (int i = 0; i < b.Width(); i++)
        {
            float sum = 0f;
            for (int j = 0; j < b.Height(); j++)
            {
                sum += v[j] * b[i, j];
            }
            destination[i] = sum;
        }

        return destination;
    }

    public static float[] Multiply(float[,] a, float[] v, float[]? destination = null)
    {
        /*          [e]
                    [f]
                  * [g]
        [ a, b, c ] [ae+bf+cg]
        [ d, e, f ] [de+ef+fg]

         */

        if (a.Width() != v.Length)
        {
            throw new Exception($"Incorrect matrix/vector size: width of a ({a.Width()}) must match length of v ({v.Length}) ");
        }

        destination ??= new float[a.Height()];
        for (int i = 0; i < a.Height(); i++)
        {
            float sum = 0f;
            for (int j = 0; j < v.Length; j++)
            {
                sum += a[j, i] * v[j];
            }
            destination[i] = sum;
        }

        return destination;
    }
    public static float[,] Multiply(float[,] a, float s, float[,]? destination = null)
    {
        destination ??= new float[a.Width(), a.Height()];
        for (int i = 0; i < a.Width(); i++)
        {
            for (int j = 0; j < a.Height(); j++)
            {
                destination[i, j] = a[i, j] * s;
            }
        }

        return destination;
    }

    public static float[,] Transpose(float[,] m, float[,]? destination = null)
    {
        destination ??= new float[m.Height(), m.Width()];

        for (int i = 0; i < m.Width(); i++)
        {
            for (int j = 0; j < m.Height(); j++)
            {
                destination[j, i] = m[i, j];
            }
        }
        return destination;
    }

    public static float[,] Multiply(float[] a, float[] b, float[,]? destination = null)
    {
        /*
          
              [ c  d  ]
        [ a ] [ ac ad ]
        [ b ] [ bc bd ]
         */

        destination ??= new float[a.Length, b.Length];
        for (int i = 0; i < a.Length; i++)
        {
            for (int j = 0; j < b.Length; j++)
            {
                destination[i, j] = a[i] * b[j];
            }
        }

        return destination;
    }

    internal static float[,] Subtract(float[,] a, float[,] b, float[,]? destination = null)
    {
        Debug.Assert(a.Width() == b.Width());
        Debug.Assert(a.Height() == b.Height());

        destination ??= new float[a.Width(), a.Height()];
        for (int i = 0; i < a.Width(); i++)
        {
            for (int j = 0; j < a.Height(); j++)
            {
                destination[i, j] = a[i, j] - b[i, j];
            }
        }
        return destination;
    }
    internal static float[,] Add(float[,] a, float[,] b, float[,]? destination = null)
    {
        Debug.Assert(a.Width() == b.Width());
        Debug.Assert(a.Height() == b.Height());

        destination ??= new float[a.Width(), a.Height()];
        for (int i = 0; i < a.Width(); i++)
        {
            for (int j = 0; j < a.Height(); j++)
            {
                destination[i, j] = a[i, j] + b[i, j];
            }
        }
        return destination;
    }
}

public static class Vector
{
    public static float[] Subtract(float[] a, float[] b, float[]? destination = null)
    {
        if (a.Length != b.Length)
        {
            throw new ArgumentException($"length of a ({a.Length}) must be the same as b ({b.Length})");
        }

        destination ??= new float[a.Length];

        for (int i = 0; i < a.Length; i++)
        {
            destination[i] = a[i] - b[i];
        }

        return destination;
    }

    public static float[] Add(float[] a, float[] b, float[]? destination = null)
    {
        if (a.Length != b.Length)
        {
            throw new ArgumentException($"length of a ({a.Length}) must be the same as b ({b.Length})");
        }

        destination ??= new float[a.Length];

        for (int i = 0; i < a.Length; i++)
        {
            destination[i] = a[i] + b[i];
        }

        return destination;
    }

    public static float[] Multiply(float[] a, float[] b, float[]? destination = null)
    {
        if (a.Length != b.Length)
        {
            throw new ArgumentException($"length of a ({a.Length}) must be the same as b ({b.Length})");
        }

        destination ??= new float[a.Length];

        for (int i = 0; i < a.Length; i++)
        {
            destination[i] = a[i] * b[i];
        }

        return destination;
    }
    public static float[] Multiply(float[] a, float b, float[]? destination = null)
    {
        destination ??= new float[a.Length];

        for (int i = 0; i < a.Length; i++)
        {
            destination[i] = a[i] * b;
        }

        return destination;
    }
}
