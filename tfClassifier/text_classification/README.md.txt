
运行程序的过程中积累一些收获！！！

1、在window Anaconda环境下安装包
方法一：打开Anaconda Prompt，pip install 包名
方法二：打开Anaconda Prompt，conda install 包名
方法三：手动下载包，放置在~\Anaconda3\Lib\site-packages目录下
2、在linux Anaconda环境下安装包
sudo pip install 包名 -d /anaconda/lib/python2.7/site-packages/。如果是压缩文件还需要解压安装。

3、python 3中，将xrange( )函数全部换为range( )即可

4、python 3中，将unichr（）函数全部转换成chr（）即可

5、在敲《Python机器学习及实践》上自然语言处理包（NLTK）上code的时候，导入nltk之后，运行出现LookupError: Recource 'tokenizers/punkt/english.pickle' not found的错误信息
1）在import nltk之后，调用之前，添加下面一句代码：nltk.download()
2）然后在弹出的“NLTK Downloader”中设置路径，点击下载即可