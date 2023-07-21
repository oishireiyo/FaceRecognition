# dlibを通してCUDAが使えない。
`face_recognition`ライブラリを使う際、正しくビルドしないとGPUを認識してくれない。
このライブラリは`dlib`をベースにしているため、`dlib`のCUDAサポートがないとCUDAを直接使用することができない。

以下のコマンドでCUDAが有効になっているか確認できる。
```bash
$ python -c "import dlib; print(dlib.DLIB_USE_CUDA)"
False
```

## dlibビルド手順
CUDAを有効にした状態でdlibをビルドする必要がある。
pipでdlibをインストールしていたら競合する可能性があるので、`pip uninstall dlib`でdlibをuninstallしてからビルドする必要がある。
ビルド手順は以下の通り。
```bash
export CUDNN_INCLUDE_DIR=/usr/local/cuda/include
export CMAKE_PREFIX_PATH=/usr/local/cuda/lib64:$CMAKE_PREFIX_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

conda activate subtitle

git clone https://github.com/davisking/dlib.git

# dlibディレクトリに移動し，ビルドディレクトリを作成
cd dlib
mkdir build
cd build

# CUDAを有効にしてdlibをコンパイル
cmake .. -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1
cmake --build .

# Pythonのラッパーをインストール
cd ..
python setup.py install

# CUDAが有効か確認
python -c "import dlib; print(dlib.DLIB_USE_CUDA)"~
```

## cuDNNがインストールされていない場合。
通常、cuDNNはCUDAと同じく`/usr/local/cuda`に置かれているので、それを前提で話を進める。
### エラー例
```
-- Found CUDA: /usr (found suitable version "10.1", minimum required is "7.5") 
-- Looking for cuDNN install...
-- *** cuDNN V5.0 OR GREATER NOT FOUND.                                                       ***
-- *** Dlib requires cuDNN V5.0 OR GREATER.  Since cuDNN is not found DLIB WILL NOT USE CUDA. ***
-- *** If you have cuDNN then set CMAKE_PREFIX_PATH to include cuDNN's folder.                ***
-- Disabling CUDA support for dlib.  DLIB WILL NOT USE CUDA
```

cuDNNがインストールされているか否かは以下の湖面で確認する。何も出力が得られない場合はインストールされていない可能性がある。
```bash
ls /usr/local/cuda/lib64 | grep libcudnn.so
```

### cuDNNインストール方法
以下のリンクにアクセス、サインアップ、サインインして`Local Installer for Linux x86_64 (Tar)`をダウンロードする。
https://developer.nvidia.com/rdp/cudnn-download \

以下で回答し、適切な場所にコピーし、パスを通す。
``` bash
tar -xf cudnn-linux-x86_64-8.9.2.26_cuda11-archive.tar.xz
cd cudnn-linux-x86_64-8.9.2.26_cuda11-archive

sudo cp include/*.h /usr/local/cuda/include
sudo cp lib64/libcudnn* /usr/local/cuda/lib64

export CUDNN_INCLUDE_DIR=/usr/local/cuda/include
export CMAKE_PREFIX_PATH=/usr/local/cuda/lib64:$CMAKE_PREFIX_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### 再度dlibのビルド。
再度dlibのビルドを行う。以下が表示されたら成功。
``` 
-- Found CUDA: /usr (found suitable version "10.1", minimum required is "7.5") 
-- Looking for cuDNN install...
-- Found cuDNN: /usr/local/cuda/lib64/libcudnn.so
-- Building a CUDA test project to see if your compiler is compatible with CUDA...
-- Building a cuDNN test project to check if you have the right version of cuDNN installed...
-- Enabling CUDA support for dlib.  DLIB WILL USE CUDA, compute capabilities: 50
```