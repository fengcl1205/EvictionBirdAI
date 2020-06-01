#/bin/bash

yum -y groupinstall "Development tools"
yum -y install zlib-devel bzip2-devel openssl-devel ncurses-devel sqlite-devel readline-devel tk-devel gdbm-devel db4-devel libpcap-devel xz-devel
mkdir /usr/local/python3
cd /usr/local/python3
yum install wget
wget https://www.python.org/ftp/python/3.5.4/Python-3.5.4.tar.xz
tar -xvJf  Python-3.5.4.tar.xz
cd Python-3.5.4
./configure --prefix=/usr/local/python3
make && make install
ln -s /usr/local/python3/bin/python3 /usr/bin/python3
ln -s /usr/local/python3/bin/pip3 /usr/bin/pip3
