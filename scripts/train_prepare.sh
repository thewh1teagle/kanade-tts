if [ -d "dataset" ]; then
    echo "Dataset already exists. Skipping download and extraction."
else
    uv run hf download thewh1teagle/hebrew-tts-dataset michael-he.7z --repo-type dataset --local-dir .
    7z x michael-he.7z
    mv michael-he dataset
fi

uv run scripts prepare_dataset.py