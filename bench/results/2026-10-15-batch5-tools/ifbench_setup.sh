#!/bin/bash
# Run INSIDE the serving container (batch 2 S3b copy, unchanged; batch 5 Q4 reuses the same pinned commit). Installs AllenAI IFBench at a pinned commit into its own venv under /tmp (the
# container layer, gone at `compose down`; the image and the serving venv are not touched), plus the NLTK data
# its verifiers tokenize with.
#   bash /tmp/ifbench_setup.sh          -> /tmp/IFBench (checkout), /tmp/ifb (venv), /tmp/nltk_data
#   then: cd /tmp/IFBench && NLTK_DATA=/tmp/nltk_data /tmp/ifb/bin/python -m run_eval \
#           --input_data=<IFBench_test.jsonl> --input_response_data=<responses.jsonl> --output_dir=<dir>
set -e
SHA=1c40f0c10d9b5c5c2f10a175a28007ebb64f7f4d   # allenai/IFBench main, `git ls-remote` 2026-10-14
cd /tmp
curl -sfL https://github.com/allenai/IFBench/archive/$SHA.tar.gz | tar xz
rm -rf IFBench && mv IFBench-$SHA IFBench
python3.12 -m venv /tmp/ifb
/tmp/ifb/bin/pip install -q --upgrade pip
/tmp/ifb/bin/pip install -q /tmp/IFBench
NLTK_DATA=/tmp/nltk_data /tmp/ifb/bin/python -c "import nltk; [nltk.download(p, download_dir='/tmp/nltk_data', quiet=True) for p in ('punkt', 'punkt_tab', 'stopwords', 'averaged_perceptron_tagger_eng')]"
echo "IFBench $SHA installed: $(/tmp/ifb/bin/pip show ifbench | grep -i ^version)"
ls -la /tmp/IFBench/data /tmp/IFBench/ifbench/data 2>/dev/null | grep -i jsonl
wc -l /tmp/IFBench/ifbench/data/IFBench_test.jsonl 2>/dev/null || wc -l /tmp/IFBench/data/IFBench_test.jsonl
