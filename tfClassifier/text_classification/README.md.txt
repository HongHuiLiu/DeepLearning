
���г���Ĺ����л���һЩ�ջ񣡣���

1����window Anaconda�����°�װ��
����һ����Anaconda Prompt��pip install ����
����������Anaconda Prompt��conda install ����
���������ֶ����ذ���������~\Anaconda3\Lib\site-packagesĿ¼��
2����linux Anaconda�����°�װ��
sudo pip install ���� -d /anaconda/lib/python2.7/site-packages/�������ѹ���ļ�����Ҫ��ѹ��װ��

3��python 3�У���xrange( )����ȫ����Ϊrange( )����

4��python 3�У���unichr��������ȫ��ת����chr��������

5�����á�Python����ѧϰ��ʵ��������Ȼ���Դ������NLTK����code��ʱ�򣬵���nltk֮�����г���LookupError: Recource 'tokenizers/punkt/english.pickle' not found�Ĵ�����Ϣ
1����import nltk֮�󣬵���֮ǰ���������һ����룺nltk.download()
2��Ȼ���ڵ����ġ�NLTK Downloader��������·����������ؼ���