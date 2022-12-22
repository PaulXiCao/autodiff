# Install
## prerequesite
* catch2 : https://github.com/catchorg/catch2
  * v3: make sure to install version 3. currently on the `devel` branch
  * in arch: catch2-git from aur
* pybind11
  * in arch: pybind11

## Build + Install
```bash
cd ~/info
git clone git@github.com:autodiff/autodiff.git
cd autodiff
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=~/info/libs/autodiff
cmake --build . --target install
```

## Example project

```bash
cp -r ~/info/autodiff/examples/cmake-project ~/info/autodiff-example-usage
cd ~/info/autodiff-example-usage
mkdir build
cd build
cmake .. -DCMAKE_PREFIX_PATH=~/info/libs/autodiff
cmake --build .
```
