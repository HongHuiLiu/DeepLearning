
Windows+Anaconda+tensorflow+keras深度学习框架搭建

1.下载安装Anaconda记住支持版本一定是python3以上的版本
链接如下：https://www.continuum.io/downloads

2.点击傻瓜式安装，下一步下一步就可安装完成，在菜单栏可以看到如下为安装完成图

3.启动Anaconda Prompt进入如下界面

4.输入conda list出现下图所示anaconda已安装的程序包

5.输入conda install tensorflow会出现如下界面

6. 在如下界面输入y会出现如下界面

7.进入python界面下输入import tensorflow若不报错说明安装成功

8.以同样的方法，输入conda install keras即可成功安装keras，遇到提示默认安装即可

9.安装完成后测试keras是否能用，import keras成功导入，且告诉我们底层用的是tensorflow


经验之谈！！！！！
执行python fully_connected_feed.py 使用Anaconda Prompt 或者cmd 命令行！！

经过多次反复实验，train部分和predict部分先注释一个再运行程序！！

怎么判断模型已训练到最优？
使用TensorBoard给出loss和准确率。

