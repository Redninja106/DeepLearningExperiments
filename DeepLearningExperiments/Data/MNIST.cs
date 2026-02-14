using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DeepLearningExperiments.Data;
internal static class MNIST
{
    private const string TrainImages = "Data/t10k-images.idx3-ubyte";
    private const string TrainLabels = "Data/t10k-labels.idx1-ubyte";

    public static float[][] Images { get; private set; }
    public static int[] Labels { get; private set; }

    public static void Load()
    {
        byte[] images = File.ReadAllBytes(TrainImages);
        uint imagesMagic = BinaryPrimitives.ReadUInt32BigEndian(images.AsSpan(0, 4));
        uint n1 = BinaryPrimitives.ReadUInt32BigEndian(images.AsSpan(4, 8));
        uint n2 = BinaryPrimitives.ReadUInt32BigEndian(images.AsSpan(8, 12));
        uint n3 = BinaryPrimitives.ReadUInt32BigEndian(images.AsSpan(12, 16));
        images = images[16..];
        Console.WriteLine($"images: {n1}x{n2}x{n3}");
        Images = new float[n1][];
        for (int i = 0; i < n1; i++)
        {
            Images[i] = new float[n2 * n3];
            for (int j = 0; j < n2; j++)
            {
                for (int k = 0; k < n3; k++)
                {
                    int idx = j * (int)n2 + k;
                    Images[i][idx] = images[i * n2 * n3 + idx] / 255f;
                }
            }
        }

        byte[] labels = File.ReadAllBytes(TrainLabels);
        uint labelsMagic = BinaryPrimitives.ReadUInt32BigEndian(labels.AsSpan(0, 4));
        uint n = BinaryPrimitives.ReadUInt32BigEndian(labels.AsSpan(4, 8));
        labels = labels[8..];
        Console.WriteLine($"labels: {n}");
        Labels = new int[n];
        for (int i = 0; i < n; i++)
        {
            Labels[i] = labels[i];
        }
    }
}
