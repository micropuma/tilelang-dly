# TileLang最简构建指南 
* clone源码  
    ```shell
    git clone --recursive git@github.com:micropuma/tilelang-dly.git
    cd tilelang
    ```

* 配置项目环境  
    ```shell
    apt-get update
    apt-get install -y python3 python3-dev python3-setuptools gcc zlib1g-dev build-essential cmake libedit-dev
    ```

    1. 启动虚拟环境  
        ```shell
        virtualenv .venv  
        source .venv/bin/activate
        ```
    2. 安装依赖  
        ```python
        pip install -r requirements-dev.txt
        ```
    3. 特别注意  
        ```python
        pip install cyton  # 不装后面源码构建 tilelang会报错
        ```


* 源码构建  
    ```shell
    mkdir build
    cd build
    cmake .. -G Ninja
    ninja
    ```

    注意：需要讲cmake中的`project(TILE_LANG C CXX)`提前，否则在cuda13下会报错。