# 阶段1: 构建应用程序
FROM python:3.10 AS builder
# 设置工作目录
WORKDIR /app
# 复制应用程序代码到镜像中
COPY . /app

#RUN apt-get install -y default-libmysqlclient-dev

# 在镜像中执行命令，安装依赖项并构建应用程序
#RUN pip install --upgrade pip
#RUN pip install ansible -i https://mirrors.aliyun.com/pypi/simple
RUN pip install --no-cache-dir -r /app/requirements.txt -i https://mirrors.aliyun.com/pypi/simple
RUN pip install opencv-python-headless -i https://mirrors.aliyun.com/pypi/simple
RUN rm -rf /app/venv
#python setup.py install
# 阶段2: 创建轻量级的运行时镜像
FROM python:3.10-slim

RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources
RUN sed -i 's/security.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources
# 设置工作目录
WORKDIR /app
# 从前一个阶段中复制构建好的应用程序
#COPY --from=builder /usr/local/bin/ansible /usr/local/bin/
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
#COPY --from=builder /app /app