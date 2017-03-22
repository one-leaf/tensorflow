本例需要最少跑1天以上，需要2M步数才有个比较好看的结果。

MAC 安装 opencv v2

```
sudo chgrp -R admin /usr/local
sudo chmod -R g+w /usr/local
brew update
brew tap homebrew/homebrew-science
brew link --overwrite jpeg
brew install opencv
cd /Library/Python/2.7/site-packages/
sudo ln -s /usr/local/Cellar/opencv/2.4.13.2/lib/python2.7/site-packages/cv.py cv.py
sudo ln -s /usr/local/Cellar/opencv/2.4.13.2/lib/python2.7/site-packages/cv2.so cv2.so
# 注意安装的版本不同，opencv 的路径也不同
```


参考来源： 

1. http://blog.topspeedsnail.com/archives/10459

2. https://github.com/DanielSlater/PyGamePlayer/tree/master/examples
