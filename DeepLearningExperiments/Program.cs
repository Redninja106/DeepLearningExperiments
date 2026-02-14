using DeepLearningExperiments;
using DeepLearningExperiments.Data;
using SimulationFramework;
using SimulationFramework.Drawing;
using SimulationFramework.Input;
using System.Numerics;

MNIST.Load();

Model m = new Model([
    new InputLayer(MNIST.Images[0].Length),
    // new DeepLayer(16, new ReLU()),
    // new DeepLayer(16, new ReLU()),
    // new DeepLayer(10, new ReLU()),

    new DeepLayer(128, new ReLU()),
    new DeepLayer(128, new ReLU()),
    new DeepLayer(128, new ReLU()),
    new DeepLayer(10, new Identity()),  // Linear activation before softmax
    new SoftmaxLayer()
    ]);

var output = m.Evaluate(MNIST.Images[0]);
Console.WriteLine(string.Join(", ", output));

float[][] labels = MNIST.Labels.Select(f => {
    var result = new float[10];
    result[f] = 1.0f;
    return result;
}).ToArray();

float[][] trainImages = MNIST.Images[..9000];
float[][] trainLabels = labels[..9000];

float[][] testImages = MNIST.Images[9000..];
float[][] testLabels = labels[9000..];

m.Train(trainImages, trainLabels, 50000, learningRate:0.001f);

Console.WriteLine("Training done... evaluating");

int correct = 0;
for (int i = 0; i < testImages.Length; i++)
{
    var result = m.Evaluate(testImages[i]);
    
    // Find predicted class (max output)
    int predicted = 0;
    for (int j = 1; j < result.Length; j++)
    {
        if (result[j] > result[predicted])
            predicted = j;
    }
    
    // Find actual class
    int actual = 0;
    for (int j = 1; j < testLabels[i].Length; j++)
    {
        if (testLabels[i][j] > testLabels[i][actual])
            actual = j;
    }
    
    if (predicted == actual)
        correct++;
}

float accuracy = (float)correct / testImages.Length * 100f;
Console.WriteLine($"Test accuracy: {accuracy:F2}% ({correct}/{testImages.Length})");

new TestApp(m).Run();

class TestApp(Model model) : Simulation
{
    float[] img = new float[28*28];

    public override void OnInitialize()
    {
    }

    public override void OnRender(ICanvas canvas)
    {
        canvas.Clear(Color.Gray);
        canvas.Translate(canvas.Width / 2f, canvas.Height / 2f);
        canvas.Scale(canvas.Height / 28f * .5f);
        canvas.Translate(-14f, -14f);

        for (int y = 0; y < 28; y++)
        {
            for (int x = 0; x < 28; x++)
            {
                float b = img[y * 28 + x];
                canvas.Fill(new ColorF(b, b, b));
                canvas.DrawRect(x, y, 1, 1);
            }
        }

        Matrix3x2.Invert(canvas.State.Transform, out var screenToTex);

        if (Keyboard.IsKeyDown(Key.Space) || Mouse.IsButtonDown(MouseButton.Left))
        {
            Vector2 xy = Vector2.Transform(Mouse.Position, screenToTex);
            int centerX = (int)xy.X;
            int centerY = (int)xy.Y;
            
            // Soft brush with Gaussian-like falloff
            for (int dy = -1; dy <= 1; dy++)
            {
                for (int dx = -1; dx <= 1; dx++)
                {
                    int px = centerX + dx;
                    int py = centerY + dy;
                    
                    if (px >= 0 && px < 28 && py >= 0 && py < 28)
                    {
                        float dist = MathF.Sqrt(dx * dx + dy * dy);
                        float intensity = MathF.Exp(-dist * dist); // Gaussian falloff
                        int idx = py * 28 + px;
                        img[idx] = MathF.Min(1f, float.Max(img[idx], intensity * .95f));
                    }
                }
            }
        }
        if (Keyboard.IsKeyDown(Key.R))
        {
            img.AsSpan().Clear();
        }

        float[] output = model.Evaluate(img);

        canvas.ResetState();
        canvas.Translate(5, 5);
        int maxIndex = 0;
        for (int i = 0; i < output.Length; i++)
        {
            canvas.Translate(0, 31);
            canvas.DrawText(i + ": " + output[i].ToString("n2"), 24f, Vector2.Zero);
            if (output[i] == output.Max())
            {
                maxIndex = i;
            }
        }

        canvas.Translate(0, 50);
        canvas.DrawText($"{maxIndex}", 50, Vector2.Zero);
    }
}