# Kaggle日記
## 20220413
Kaggleをはじめた。
最初はTabular Playground Series - Apr 2022に挑戦する。
他の人のコードを見るところから始める。

## 20220414
EDAのコードを一つ写経してみたが、そこからモデリングに移ったときに何をすれば良いのか全く分からない。  
他のモデリング(LightGBM)を行っているKernelを見つけたので明日までに全て写経してみようと思う。  
また、とりあえずkaggleのslackに入ってみた。今後質問できそうだったら質問してみる。

## 20220415
LightGBMのコードを写経して提出まで行ったが、いまいち何をやっているのか分からない。  
Kaggleで勝つ本をサラッと読んでみたが、様々なモデルを紹介してくれているものの、コードの内容が分からないので、現状の解決には至らなさそう。  
  
https://www.kaggle.com/code/kotanakajima/kaggle-prediction-house-prices/edit  
House Priceのこのkernelを用いてコードを理解しながら流れをつかもうと思う。あとはpandasの勉強が必要そう。。。

## 20220416
LightGBMにおいて必要な特徴量エンジニアリング代表例  
①特徴量同士の四則演算  
②集約特徴量(特徴量のmean/median/std/max/min)  
↑＋特徴量の削減(学習時間短縮、メモリ節約のため)を行えばとりあえずbaselineのサブミットはできそう。

## 20220417
Tabular Playground Series - Feb 2022をやってみようとしてKernelをみているが、ドメイン知識をフル活用していて、EDAが難しい。  
とりあえず難しいEDAをやっていないものを見つけてsubmissionまでなるべく独力でやってみる。

## 20220418
Tabular Playground Series - Feb 2022  
https://www.kaggle.com/code/odins0n/tps-feb-22-eda-modelling  
このKernelが分かりやすそうなので手を付けている。

また、以下のyoutubeがKaggleの進め方の理解に役立ったのでメモしておく。  
https://www.youtube.com/watch?v=Ug5uce0kbtQ  

## 20220419
Google colabからkaggleをできるようにした。下の2リンクを参照。  
https://qiita.com/takeru0208/items/3bc89dbfe25d2de600dc  
https://dreamer-uma.com/kaggle-api-colab/  
ダウンロードしたkaggleのファイルにアクセスするため、毎回以下のコードを実行してGoogleDriveをマウントする必要がある。
```python
from google.colab import drive  
drive.mount('/gdrive')
```
### sns.kdeplot
Tabular Playground Series - Feb 2022  
読んでいるKernelでKDE(Kernel Density Estimation, カーネル密度推定)のグラフをたくさん生成している。  
簡単には、標本から全体の分布を推定する際、標本点を正規分布などのカーネル関数で表現し、足し合わせることで分布を算出する手法らしい。  
https://club.informatix.co.jp/?p=1176

## 20220420
StratifiedKFold(層化K分割交差検証)について。要約すると、訓練データ内の各ラベルと検証データ内の各ラベルの比率を一定にしたKFold(あるラベルが訓練データに10個、検証データに0個のようなことを防いでくれる)。
https://xn--stanalytics-note-5x3o0cry2m2n.xyz/machine-learning/stratified-kfolds-cross-validator/  
  
LightGBMでGPUを使う方法は以下。  
https://www.guruguru.science/competitions/13/discussions/64f95387-97b8-4c49-9eb1-8a0e12ed4469/
