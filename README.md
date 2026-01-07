


How to use Jutyper Notebook on the BW UniCluster 3.0:
    1. activate your conda environment
    1. conda install -y ipykernel
    2. start salloc session z. B.: salloc --partition=dev_gpu_h100 --ntasks=1 --time=30 --mem=5000 --gres=gpu:1
    3. register kernel: python -m ipykernel install --user --name llm-env --display-name "Python (llm-env)"
    4. select the new kernel in your ipynb
