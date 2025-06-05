CONDA ENVIRONMENT CREATION:

1. conda create -n deepsvdd python=3.8 -y
2. conda activate deepsvdd
3. Before installing requirements install pytorch:
      pip install torch==2.3.0+cu118 torchvision==0.18.0+cu118 torchaudio==2.3.0 --extra-index-url https://download.pytorch.org/whl/cu118
4. Install updated list of newer packages from the requirements.txt file:
      pip install -r requirements.txt

> **NOTE:** If you want to run our adjusted DeepSVDD code, please activate <deepsvdd> conda environment! One cannot use conda environment 
> of the Diffusion-Models-Embedding-Space-Defense repository.

RUNNING main.py:

- run the code for the first time to pre-train mlp_autoencoder on the src/main.py path:
python main.py visu mlp ../log ../data --objective one-class --pretrain True --n_epochs 50 --lr 0.001 --batch_size 128 --device cuda --ae_n_epochs 100 --ae_lr 0.001 --ae_batch_size 128


- run the code when mlp_autoencoder is already trained:
python main.py visu mlp ../log ../data --objective one-class --pretrain False --load_model C:\Users\Asus\PycharmProjects\Deep-SVDD-PyTorch\log\model.tar --n_epochs 50 --lr 0.001 --batch_size 128 --device cuda --ae_n_epochs 100 --ae_lr 0.001 --ae_batch_size 128