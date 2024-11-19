echo "# HunyuanGraph_SC25_0.0.0" > README.md
git init
# 添加所有文件到暂存区
git add .
# 描述更改内容
git commit -m "first commit"
git branch -M main
# 移除旧的远程仓库
git remote remove origin
git remote add origin git@github.com:Elegancejiang/HunyuanGraph_SC25_0.0.0.git
git push -u origin main