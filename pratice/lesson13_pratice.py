from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Input,Dense
from keras.layers import GlobalAveragePooling2D
# ------------------------------------------------------------------------------------------------------------------- #
import os   # 去除bug
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# ------------------------------------------------------------------------------------------------------------------- #
# 此章節介紹池化層
# Max Pooling :
# 最大池化為選擇 Kernel 內最大的值，其用意在於選取特徵，
# 保留重要紋理，並降低過擬合( Overfitting ) 。而 Max Pooling 在某種程度上也能提升圖像旋轉、平移、縮放的不變性。
# Average Pooling :
# 相較於Max Pooling用來提取重要特徵與邊緣，Average Pooling強調的是特徵的平滑性，然而其缺點在於不管重要或不重要特徵都平均計算。
# ------------------------------------------------------------------------------------------------------------------- #
# 一方面來說我們透過 Pooling 降低 Feature Maps 的尺度，藉此降低運算量，提取特徵並加速收斂。
# 但就另一方面而言，我們同時也會失去部分特徵值，因此是否使用 Pooling 仍有爭議，就我們的實驗結果來看，
# 一兩層的 Pooling 確實能增快收斂並達到一樣或更好的準度(相較於沒有使用Pooling 的模型)，
# 然而大量的 Pooling 雖然收斂更快，最後準度卻也比較低，大家可以自己去嘗試看看結果如何。
# ------------------------------------------------------------------------------------------------------------------- #
# 全連接層( FC ) :
# 既然有了 CNN，為什麼後面還要接上 Fully Connected 層呢？
# 「其目的主要是要是為了利用全連接層的神經元做為分類器各類別的代表機率。」
# 一般的全連接層主要分為 輸入層、隱藏層、輸出層，而在這裡我們的輸入層就是攤平的CNN層，輸出層看是幾類的分類就用幾個神經元
# ------------------------------------------------------------------------------------------------------------------- #
# 攤平( Flatten ):
# 攤平其實就是將大於 1 維的 Tensor 拉平為二維( batch size, channels )，通常經過 CNN 的 Feature Map 為四維，
# 拉平為二維才能夠與全連接層合併。
# 而除了 Flatten 以外，Global Average Pooling 也是常見連結 CNN 與 FC 的方式 。
# ------------------------------------------------------------------------------------------------------------------- #
# ---------------------------------作業: 運用Pooling 與Flatten觀看輸出Tensor的尺寸------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------- #
# 一般API在跑張量(tensor)，通常是一個批次量的資料在執行
# 假設有100筆資料，每個資料都是10*10的彩色圖片(所以每個資料都是3張圖片，分別為R、G、B三張)，
# 此時的輸入資料的大小就是100 ×10 ×10 ×3，解讀方式batch ×height × width × channel
# ------------------------------------------------------------------------------------------------------------------- #
input_shape = (32, 32, 3)
model = Sequential()
# ------------------------------------------------------------------------------------------------------------------- #
# Pro1 : pooling_size=2,2 strides=2,2 輸出feature map 大小為多少？
# ------------------------------------------------------------------------------------------------------------------- #
print('Pro-1')
# pool_size :整數，最大池化的窗口大小。
# strides :整數，或者是None。作為縮小比例的因數。例如，2會使得輸入張量縮小一半。如果是None，那麼默認值是pool_size。
model.add(Conv2D(32, kernel_size=(3, 3), padding='same',input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
model.summary()
print('--------------------------------------------------------------------------------------------------------------')
# ------------------------------------------------------------------------------------------------------------------- #
# Pro2 : pooling_size=2,2 strides=2,2 輸出feature map 大小為多少？
# ------------------------------------------------------------------------------------------------------------------- #
print('Pro-2')
model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
model.summary()
print('--------------------------------------------------------------------------------------------------------------')

# ------------------------------------------------------------------------------------------------------------------- #
# Pro3 : pooling_size=1,1 strides=1,1 輸出feature map 大小為多少？
# ------------------------------------------------------------------------------------------------------------------- #
print('Pro-3')
model.add(Conv2D(128, kernel_size=(3, 3), padding='same'))
model.add(MaxPooling2D(pool_size=(1, 1),strides=(1,1)))
model.summary()
print('--------------------------------------------------------------------------------------------------------------')

# ------------------------------------------------------------------------------------------------------------------- #
# Pro4 : Flatten完尺寸如何變化？
# ------------------------------------------------------------------------------------------------------------------- #
print('Pro-4')
model.add(Conv2D(10, kernel_size=(3, 3), padding='same'))
model.add(Flatten())
model.summary()
print('--------------------------------------------------------------------------------------------------------------')

# ------------------------------------------------------------------------------------------------------------------- #
# Pro5 : 關掉Flatten，使用GlobalAveragePooling2D，完尺寸如何變化？
# Global Average Pooling (GAP) 就是將每張Feature Map上的資訊以平均的方式壓為一個值
# ------------------------------------------------------------------------------------------------------------------- #
classifier=Sequential()
input_shape = (32, 32, 3)
print('Pro-5')
classifier.add(Conv2D(10, kernel_size=(3, 3), padding='same',input_shape=input_shape))
classifier.add(GlobalAveragePooling2D())
classifier.summary()
print('--------------------------------------------------------------------------------------------------------------')

# ------------------------------------------------------------------------------------------------------------------- #
# Pro6 : 全連接層使用28個units
# 近期文獻偏好使用越少FC層越好，主要是由於FC層容易產生大量參數，
# FC層的最後一層要使用與預分類類別一樣多的神經元，當作各個類別的輸出特徵值，並透過Softmax轉換成機率值。
# ------------------------------------------------------------------------------------------------------------------- #
print('Pro-6')
model.add(Dense(28))
model.summary()
print('--------------------------------------------------------------------------------------------------------------')

# ---------------------------------------------------------------------------------------------- #
# 補充:
# 1.CNN 主要借助卷積層(Convolution Layer)的方法，將Input從原始的點陣圖，改為經過影像處理技術萃取的特徵，
# 等於是提供更有效的資訊給模型使用，因此，預測的效果就會顯著的向上提升。
# 厲害的是，我們也不用撰寫影像處理的程式碼，Neural Network(也許是框架?) 會幫我們自動找出這些特徵。
# ---------------------------------------------------------------------------------------------- #
# 2.卷積層是作Input的改善，所以，它通常會放在模型的『前』幾層，後面會緊跟著池化層，以簡化計算。
# ---------------------------------------------------------------------------------------------- #
# 3.最後，模型會回歸到全連接層(Dense)進行分類，卷積層是多維的陣列，全連接層的Input通常為一維，
# 中間維度的轉換就靠『扁平層』(Flatten)來處理。
# ---------------------------------------------------------------------------------------------- #
# 4.為避免『過度擬合』(overfit)，我們會在特定的地方加上Dropout層，來防止過度擬合。
# ---------------------------------------------------------------------------------------------- #








