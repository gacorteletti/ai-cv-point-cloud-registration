## For classical methods(ICP, RANSAC and FPFH)

conda create --name pcr python=3.11
conda activate pcr
python --version

conda install pandas
conda install numpy

conda install pip
pip install open3d

numpy                     2.2.2           py311h5d046bc_0    conda-forge
open3d                    0.19.0                   pypi_0    pypi
pandas                    2.2.3           py311h7db5c69_1    conda-forge



## For FCGF (AI)

nvidia-smi --> CUDA Version: 12.2 (top right of the table)

conda create -n fcgf python=3.9
conda activate fcgf
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

conda install -c conda-forge openblas
find $CONDA_PREFIX -name cblas.h
    [should return something like: `/home/corteletti/miniconda3/envs/fcgf/include/cblas.h`]
    [else, run: `export C_INCLUDE_PATH=$CONDA_PREFIX/include:$C_INCLUDE_PATH`]

find $CONDA_PREFIX/lib -name "libopenblas.so*"
    [should return: `/home/corteletti/miniconda3/envs/fcgf/lib/libopenblas.so.0`
               and: `/home/corteletti/miniconda3/envs/fcgf/lib/libopenblas.s`]
    [else, if no file libopenblas.so.0 then run:`ln -s $CONDA_PREFIX/lib/libopenblas.so.0 $CONDA_PREFIX/lib/libopenblas.so`]
        [if there was already that file, running this will return: `failed to create symbolic link '/home/corteletti/miniconda3/envs/fcgf/lib/libopenblas.so': File exist`]

ls -l $CONDA_PREFIX/lib/libopenblas.so
    [should return: `lrwxrwxrwx 1 corteletti corteletti 23 Feb 20 22:13 /home/corteletti/miniconda3/envs/fcgf/lib/libopenblas.so -> libopenblasp-r0.3.29.so`]

export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

pip install MinkowskiEngine==0.5.4

[to test it, run: `conda install ipykernel`then open a python script and run:
    `import MinkowskiEngine as ME`
    `print(ME.__version__)`
this should return:
    `0.5.4`]