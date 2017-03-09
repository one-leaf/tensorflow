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


参考来源： http://blog.topspeedsnail.com/archives/10459