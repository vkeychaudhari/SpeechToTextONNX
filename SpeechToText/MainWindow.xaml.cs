using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NAudio.Wave;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Threading.Tasks;
using System.Windows;

namespace SpeechToText
{
    public partial class MainWindow : Window
    {
        private WaveInEvent _waveSource;
        private WaveFileWriter _waveFile;
        private string _tempAudioFile = "temp_teacher_voice.wav";

        public MainWindow()
        {
            InitializeComponent();
        }

        // UI var runtime logs dakhavnyasathi ha special function
        private void LogToUI(string message)
        {
            Dispatcher.Invoke(() =>
            {
                TxtResult.Text += $"[{DateTime.Now:HH:mm:ss}] {message}\r\n";
                // Auto scroll to bottom mhanje navin logs khali distil
                TxtResult.ScrollToEnd();
            });
        }

        private void BtnRecord_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                BtnRecord.IsEnabled = false;
                BtnStop.IsEnabled = true;
                TxtResult.Text = ""; // Juni mahiti clear karne

                LogToUI("Mic suru hot aahe...");

                _waveSource = new WaveInEvent();
                _waveSource.WaveFormat = new WaveFormat(16000, 1); // 16kHz Mono

                _waveSource.DataAvailable += WaveSource_DataAvailable;
                _waveSource.RecordingStopped += WaveSource_RecordingStopped;

                _waveFile = new WaveFileWriter(_tempAudioFile, _waveSource.WaveFormat);
                _waveSource.StartRecording();

                LogToUI("Recording suru zhale... Aata bola.");
            }
            catch (Exception ex)
            {
                LogToUI($"[ERROR RECORD START]: {ex.Message}");
                BtnRecord.IsEnabled = true;
                BtnStop.IsEnabled = false;
            }
        }

        private void WaveSource_DataAvailable(object sender, WaveInEventArgs e)
        {
            if (_waveFile != null)
            {
                _waveFile.Write(e.Buffer, 0, e.BytesRecorded);
                _waveFile.Flush();
            }
        }

        private void BtnStop_Click(object sender, RoutedEventArgs e)
        {
            BtnStop.IsEnabled = false;
            LogToUI("Recording थांबवत आहे...");
            _waveSource?.StopRecording();
        }

        private async void WaveSource_RecordingStopped(object sender, StoppedEventArgs e)
        {
            // Clean up audio resources
            _waveSource?.Dispose();
            _waveSource = null;
            _waveFile?.Dispose();
            _waveFile = null;

            LogToUI("Recording purna zhale. Audio processing background madhe patvle.");

            // Task.Run vaparun background thread suru kela jyane UI freeze honar nahi
            string finalResult = await Task.Run(() =>
            {
                return RunMoonshineModelInference(_tempAudioFile);
            });

            LogToUI($"--- FINAL RESULT ---\r\n{finalResult}");
            BtnRecord.IsEnabled = true;
        }

        // Actual ONNX execution aani step-by-step logging
        private string RunMoonshineModelInference(string audioFilePath)
        {
            try
            {
                string encoderPath = "encoder_model.onnx";
                string decoderPath = "decoder_model_merged.onnx";

                // 1. Files check karne
                if (!File.Exists(encoderPath) || !File.Exists(decoderPath))
                {
                    LogToUI("[ERROR]: ONNX files bin folder madhe sapdlya नाहीत!");
                    return "Model Files Missing";
                }

                LogToUI("Audio file loading suru...");
                float[] audioRawData = LoadAndResampleAudio(audioFilePath);
                LogToUI($"Audio raw data loaded. Total samples: {audioRawData.Length}");

                // 2. ONNX Sessions Load karne
                LogToUI("ONNX Inference Sessions initialize hot aahet (Yala thoda vel lagu shakto)...");
                using var encoderSession = new InferenceSession(encoderPath);
                using var decoderSession = new InferenceSession(decoderPath);
                LogToUI("Both ONNX Models Successfully Loaded into memory!");

                // 3. Encoder execution
                LogToUI("Encoder model run hot aahe...");

                // मॉडेलला स्वतःला विचारणे की त्याच्या इनपुटचे खरे नाव काय आहे
                string actualEncoderInputName = new List<string>(encoderSession.InputMetadata.Keys)[0];
                LogToUI($"Encoder ला हे नाव अपेक्षित आहे: '{actualEncoderInputName}'");

                int[] audioShape = new int[] { 1, audioRawData.Length };
                var audioTensor = new DenseTensor<float>(audioRawData, audioShape);

                var encoderInputs = new List<NamedOnnxValue>
                {
                    // "audio_pcm" ऐवजी मॉडेलमधून आलेले डायनॅमिक नाव वापरणे
                    NamedOnnxValue.CreateFromTensor(actualEncoderInputName, audioTensor)
                };

                using var encoderOutputs = encoderSession.Run(encoderInputs);
                var audioEmbeddings = encoderOutputs[0].AsTensor<float>();
                LogToUI("Encoder processing complete. Audio embeddings generated.");

                // पुढच्या स्टेपमध्ये (Decoder) सुद्धा नाव चुकू नये म्हणून आपण आधीच त्याचे नावे प्रिंट करत आहोत:
                var expectedDecoderInputs = new List<string>(decoderSession.InputMetadata.Keys);
                LogToUI($"Decoder ला अपेक्षित इनपुट्स: {string.Join(", ", expectedDecoderInputs)}");

                // 4. Decoder Loop (Growing Sequence Approach)
                LogToUI("Speech to Text Decoding सुरू होत आहे...");

                var tokenDictionary = LoadTokenizer();
                List<long> tokensSequence = new List<long> { 1 }; // Start token
                bool stop = false;
                string finalSpokenText = "";

                while (!stop && tokensSequence.Count < 100)
                {
                    int[] tokensShape = new int[] { 1, tokensSequence.Count };
                    var tokensTensor = new DenseTensor<long>(tokensSequence.ToArray(), tokensShape);

                    bool[] useCacheData = new bool[] { false };
                    var useCacheTensor = new DenseTensor<bool>(useCacheData, new int[] { 1 });

                    var decoderInputs = new List<NamedOnnxValue>
                    {
                        NamedOnnxValue.CreateFromTensor("input_ids", tokensTensor),
                        NamedOnnxValue.CreateFromTensor("encoder_hidden_states", audioEmbeddings),
                        NamedOnnxValue.CreateFromTensor("use_cache_branch", useCacheTensor)
                    };

                    using var decoderOutputs = decoderSession.Run(decoderInputs);
                    var logits = decoderOutputs[0].AsTensor<float>();

                    // नवीन आणि एकदम अचूक GetBestToken कॉल करणे
                    long nextToken = GetBestToken(logits);

                    LogToUI($"[Debug] AI ने शोधलेला टोकन: {nextToken}"); // हा लॉग आपल्याला लपलेला टोकन दाखवेल

                    if (nextToken == 2 || nextToken == 0)
                    {
                        LogToUI("[Info] मॉडेलने वाक्य संपल्याचा सिग्नल दिला (End of Speech).");
                        stop = true;
                    }
                    else
                    {
                        tokensSequence.Add(nextToken);

                        if (tokenDictionary.TryGetValue(nextToken, out string decodedWord))
                        {
                            // AI चा 'स्पेस' सिम्बॉल काढून तिथे खरी स्पेस टाकणे
                            decodedWord = decodedWord.Replace("▁", " ").Replace(" ", " ");

                            finalSpokenText += decodedWord;
                            LogToUI($"लाईव्ह शब्द: {decodedWord}");
                        }
                    }
                }

                LogToUI($"\n--- फायनल टेक्स्ट ---\n{finalSpokenText.Trim()}");
                return finalSpokenText.Trim();
            }
            catch (Exception ex)
            {
                LogToUI($"[CRITICAL RUNTIME ERROR]: {ex.Message}\nStack Trace: {ex.StackTrace}");
                return $"Error: {ex.Message}";
            }
        }

        private float[] LoadAndResampleAudio(string wavFilePath)
        {
            var samples = new List<float>();
            using (var reader = new AudioFileReader(wavFilePath))
            {
                float[] buffer = new float[4096];
                int read;
                while ((read = reader.Read(buffer, 0, buffer.Length)) > 0)
                {
                    for (int i = 0; i < read; i++) samples.Add(buffer[i]);
                }
            }
            return samples.ToArray();
        }

        private long GetBestToken(Tensor<float> logits)
        {
            // मॉडेलने दिलेल्या आउटपुटची साईज (Vocab Size) शोधणे
            int vocabSize = logits.Dimensions[logits.Dimensions.Length - 1];

            // नेहमी 'शेवटच्या' टोकनचे प्रेडिक्शन तपासणे
            int startIndex = (int)logits.Length - vocabSize;

            float maxVal = float.MinValue;
            long bestIndex = 0;

            for (int i = 0; i < vocabSize; i++)
            {
                float val = logits.GetValue(startIndex + i);
                if (val > maxVal)
                {
                    maxVal = val;
                    bestIndex = i;
                }
            }
            return bestIndex;
        }

        private Dictionary<long, string> LoadTokenizer()
        {
            var vocabDict = new Dictionary<long, string>();
            try
            {
                string jsonContent = File.ReadAllText("tokenizer.json");
                using var document = JsonDocument.Parse(jsonContent);
                var vocab = document.RootElement.GetProperty("model").GetProperty("vocab");

                foreach (var element in vocab.EnumerateObject())
                {
                    // AI च्या डिक्शनरीमध्ये स्पेससाठी 'Ġ' वापरतात, आपण तो काढून स्पेस टाकू
                    string word = element.Name.Replace("Ġ", " ");
                    word = word.Replace("Ċ", "\n"); // नवीन लाईनसाठी
                    long id = element.Value.GetInt64();
                    vocabDict[id] = word;
                }
                LogToUI("Tokenizer (शब्दकोश) यशस्वीरित्या लोड झाला!");
            }
            catch (Exception ex)
            {
                LogToUI($"[Warning] Tokenizer लोड झाला नाही: {ex.Message}");
            }
            return vocabDict;
        }
    }
}