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
  
LightGBMでGPUを使う方法は以下(できないことがあったのはColabのGPUが使用制限で使えなかったから？)。  
https://datadriven-rnd.com/googlecolab-gpu/  
https://qiita.com/DS27/items/4ce48f08ad3ffb7d9a92

##  20220421
LightGBMのモデルを写経して実行したものの、結局次のステップが全く分からないので、Udemyのデータ分析の講座を受けることにする。  
Udemyが終わるまでKaggleは休止。

## 20220422
Dockerのpullをwindowsで行う際に、通常のやり方だと正しくマウントできなかったのでメモ。  
https://pickerlab.net/2021/05/02/you-shared-a-windows-file-into-a-wsl-2-container/  
https://zenn.dev/kathmandu/articles/4a86c3d75b93c3  
https://zenn.dev/kathmandu/articles/4a86c3d75b93c3

## 20220524
Pythonの勉強、統計検定の勉強をしていたら（＋Kaggle何すれば良いか分からないと言っていたら）一か月経ってしまった。  
とりあえず何回かPlay groundで一通りのモデル作成(LGBM, XGBoost, NN)、パラメータチューニングなどやってみた後は画像コンペに取り組もうと思う。  
LightGBMのパラメータチューニングに関する良記事があったので掲載する。  
https://knknkn.hatenablog.com/entry/2021/06/29/125226  
  
lightgbmのearly_stopping_roundsをcallback引数で指定する方法。  
https://qiita.com/c60evaporator/items/2b7a2820d575e212bcf4

## 20220526
Optunaでパラメータの最適化を行ってみた。時間がかかりすぎてなんだこれってなってる。  
LightGBM用のTunerもあるが、一部のパラメータについてのみ最適化が可能。  
https://tech.preferred.jp/ja/blog/hyperparameter-tuning-with-optuna-integration-lightgbm-tuner/
結局今のところは手動でパラメータチューニングするのが早そう。

## 20220527
本日もパラメータチューニングに時間をかけていた。  
パラメータは特徴量などが変わると最適な値が変わってしまう&非常に時間がかかるので、コンペの最後の方にチューニングした方が良いことを理解した。  
  
パラメータチューニングについて異常に細かく解説している記事があったので、実力が付いたら読み直したい。    
https://qiita.com/c60evaporator/items/ca7eb70e1508d2ba5359  

また、LightGBMでGPUを使っているが、Kaggle NotebookでGPUの使用率が3%程度までしか行かなかったので、正しくできていないのかもしれない。CPUとあまり処理速度に差が無いように感じるので、無理にGPUを使う必要はないかも。
