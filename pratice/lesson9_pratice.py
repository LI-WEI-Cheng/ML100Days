#
# 了解計算圖片的導數就會取得邊緣，在電腦視覺中我們也可以把這個過程稱為「抽取特徵」
# 直覺的解釋為「圖片中最特別的地方」，可以是邊緣，輪廓，紋理等資訊
# SIFT 就是其中一種表徵 (appearance feature)，基於局部的外觀特徵，進一步考慮到圖片特徵的狀況 :
# 縮放不變性，旋轉不變性
# 光線與雜訊容忍度高
# -------------------------------------------------------------------------------------------------------------------- #
# https://www.itread01.com/content/1541874126.html   --> 去看
# -------------------------------------------------------------------------------------------------------------------- #
# Goal :
# 尺度空間理論 --> 尺度空間中各尺度影象的模糊程度逐漸變大，能夠模擬人在距離目標由近到遠時目標在視網膜上的形成過程，尺度越大影象越模糊。
# 在 SIFT 演算法中可以了解到如何做關鍵點偵測
# 並如何抽出 SIFT 特徵來敘述關鍵點
# -------------------------------------------------------------------------------------------------------------------- #
# SIFT 特徵 - 尺度不變性 :
# SIFT 主要考慮到的問題之一是尺度
# 用機器視覺系統分析未知場景時，計算機並不預先知道影象中物體的尺度。我們需要同時考慮影象在多尺度下的描述，獲知感興趣物體的最佳尺度。
# 另外如果不同的尺度下都有同樣的關鍵點，那麼在不同的尺度的輸入影象下就都可以檢測出來關鍵點匹配，也就是尺度不變性。
# 影象的尺度空間表達就是影象在所有尺度下的描述。
# 以 corner detector (e.g. Harris) 為例，Filter 可以偵測到範圍以內的角落點，但是同樣的 pattern 放大後以同樣的 Filter 去偵測就會失敗
# 不同的尺度下都有同樣的關鍵點，那麼在不同的尺度的輸入影象下就都可以檢測出來關鍵點匹配，也就是尺度不變性。
# -------------------------------------------------------------------------------------------------------------------- #
# SIFT 特徵 - 尺度空間極值偵測 :
# SIFT 會基於邊緣檢測抽取特徵，但不是使用前面提過的 Sobel 。概念上是 LoG 但是考慮到計算量使用 DoG 做邊緣檢測
# -------------------------------------------------------------------- #
# 尺度空間表達與金字塔多解析度表達 :
# 高斯模糊
# --> 一個影象的尺度空間L(x,y,σ) ,定義為原始影象I(x,y)與一個可變尺度的2維高斯函式G(x,y,σ)卷積運算。
#     尺度是自然客觀存在的，不是主觀創造的。高斯卷積只是表現尺度空間的一種形式。
#     高斯模版是圓對稱的，且卷積的結果使原始畫素值有最大的權重，距離中心越遠的相鄰畫素值權重也越小。
#     在實際應用中，在計算高斯函式的離散近似時，在大概3σ距離之外的畫素都可以看作不起作用，這些畫素的計算也就可以忽略。
#     所以，通常程式只計算(6σ+1)*(6σ+1)就可以保證相關畫素影響。
#     高斯模糊另一個很厲害的性質就是線性可分：使用二維矩陣變換的高斯模糊可以通過在水平和豎直方向各進行一維高斯矩陣變換相加得到。
# -------------------------------------------------------------------- #
# 金字塔多解析度
# --> 金字塔是早期影象多尺度的表示形式。影象金字塔化一般包括兩個步驟：使用低通濾波器平滑影象；
#     對平滑影象進行降取樣（通常是水平，豎直方向1/2），從而得到一系列尺寸縮小的影象。
# -------------------------------------------------------------------- #
# 尺度空間表達和金字塔多解析度表達之間最大的不同是：
# 尺度空間表達是由不同高斯核平滑卷積得到，在所有尺度上有相同的解析度；
# 而金字塔多解析度表達每層解析度減少固定比率。
# 所以，金字塔多解析度生成較快，且佔用儲存空間少；而多尺度表達隨著尺度引數的增加冗餘資訊也變多。
# 多尺度表達的優點在於影象的區域性特徵可以用簡單的形式在不同尺度上描述；而金字塔表達沒有理論基礎，難以分析影象區域性特徵。
# -------------------------------------------------------------------- #
# Laplacian of Gaussian (LoG) (高斯拉普拉斯LoG金字塔) :
# 結合尺度空間表達和金字塔多解析度表達，就是在使用尺度空間時使用金字塔表示，也就是計算機視覺中最有名的拉普拉斯金子塔
# Ans --> 先對圖片做 Gaussian Blur 再算二階導數取得邊緣
# -------------------------------------------------------------------- #
# Difference of Gaussian (DoG)
# DoG（Difference of Gaussian）其實是對高斯拉普拉斯LoG的近似，也就是對的近似。
# SIFT演算法建議，在某一尺度上的特徵檢測可以通過對兩個相鄰高斯尺度空間的影象相減，得到DoG的響應值影象D(x,y,σ)。.
# 然後仿照LoG方法，通過對響應值影象D(x,y,σ)進行區域性最大值搜尋，在空間位置和尺度空間定位區域性特徵點。
# Ans --> 圖片經過不同程度的縮放後計算出不同程度的 Gaussian Blur 最後合併得到一個 Gaussian Pyramid，其差值即為 DoG
#         結果可以視為 LoG 的約略值 (沒有做二階導數)
# -------------------------------------------------------------------- #
# 這邊討論的特徵主要是物體的邊緣
# 而二階導數是個適合的工具來找出邊緣，因此這邊才會以此討論 LoG 與 DoG
# -------------------------------------------------------------------- #
# 金字塔構建 :
# 為了得到DoG影象，先要構造高斯金字塔。
# 高斯金字塔在多解析度金字塔簡單降取樣基礎上加了高斯濾波，也就是對金字塔每層影象用不同引數的σ做高斯模糊，使得每層金字塔有多張高斯模糊影象。
# 金字塔每層多張影象合稱為一組（Octave），每組有多張（也叫層Interval）影象。
# 另外，降取樣時，金字塔上邊一組影象的第一張影象（最底層的一張）是由前一組（金字塔下面一組）影象的倒數第三張隔點取樣得到。
# [ σ為尺度空間座標 ]
# 構建高斯金字塔之後，就是用金字塔相鄰影象相減構造DoG金字塔。
# -------------------------------------------------------------------- #
# SIFT 特徵 - 尺度空間極值偵測 (DoG 尺度) :
# 圖片的一種 scale 稱為一個 octave
# 每種 scale 的圖片經過不同程度的 Gaussian Blur 在計算其差值
# 最後會得到下圖最後的 DoG (Gaussian Pyramid)
# -------------------------------------------------------------------------------------------------------------------- #
# SIFT 特徵 - 尺度空間極值偵測 (極值偵測) :
# 前面提到的 DoG 影像包含多種尺度
# 接著要針對每個 pixel 判斷是否為極值
# 判斷範圍 8+9*2 = 26(立體的周圍 --> 有上下兩層)
# 自己本身周遭的 8 個 pixel
# 同一個 scale 圖片但不同模糊尺度 相鄰位置共 9*2=18 個 pixel
# 假如該 pixel 為判斷範圍內的最大 / 最小值
# 則將其設為有興趣的關鍵點(一個點如果在DOG尺度空間本層以及上下兩層的26個領域中是最大或最小值時，就認為該點是圖像在該尺度下的一個特徵點)
# 在極值比較的過程中，每一組圖像的首末兩層是無法進行極值比較的，為了滿足尺度變化的連續性，我們在每一組圖像的頂層繼續用高斯模糊生成了3幅圖像，高斯金字塔有每組S+3層圖像。
# DOG金字塔每組有S+2層圖像. 上圖所示S為3.
# -------------------------------------------------------------------------------------------------------------------- #
# SIFT 特徵 - 關鍵點定位:
# 離散空間的極值點並不是真正的極值點。
# 利用已知的離散空間點插值得到的連續空間極值點的方法叫做子像素插值（Sub-pixel Interpolation）。
# 差值中心偏移量大於0.5時（即x或y或），意味著插值中心已經偏移到它的鄰近點上，所以必須改變當前關鍵點的位置。同時在新的位置上反复插值直到收斂；
# 過小的點易受噪聲的干擾而變得不穩定，所以將小於某個經驗值(Lowe論文中使用0.03，Rob Hess等人實現時使用0.04/S)的極值點刪除。
# 同時，在此過程中獲取特徵點的精確位置(原位置加上擬合的偏移量)以及尺度σ(o,s)。
# -------------------------------------------------------------------- #
# 經過多尺度極值偵測之後，會得到許多候選的關鍵點，其中也包含許多噪音跟邊的關鍵點，需要更進一步根據周遭資訊來修正並過濾關鍵點 :
# --> 鄰近資料差補 : 主要根據相鄰資訊來修正極值的位置
# --> 過濾不明顯關鍵點 : 根據計算曲率來判斷是否為不明顯的關鍵點
# --> 過濾邊緣關鍵點 : 根據計算曲率來判斷是否為不明顯的關鍵點
# -------------------------------------------------------------------------------------------------------------------- #
# SIFT 特徵 - 方位定向 :
# 前面我們定義並過濾了許多關鍵點，但是關鍵點只有包含尺度跟位置
# SIFT 還想要保有旋轉不變性，因此要給關鍵點定義一個方向
# --> 以每 10 度為單位計算周圍的梯度值
# --> 梯度值最大的方向當作是該關鍵點的主要方向
# (補充)
# 算法流程：
# 1. 遍歷特徵點集合points，搜索每個特徵點的鄰域，半徑為rad，生成含有36柱的方向直方圖，梯度直方圖範圍0~360度，其中每10度一個柱。
# 2. 利用高斯加權對方向直方圖進行兩次平滑，增加穩定性
# 3. 通過峰值比較，求取關鍵點方向（可能是多個方向）；
# 4. 通過Taylor展開式對上述峰值進行二次曲線擬合，計算關鍵點精確方向，即重新計算峰值所在bin的值；
# 5. 根據bin的值還原角度，作為特徵點的方向。
# -------------------------------------------------------------------------------------------------------------------- #
# SIFT 特徵 - 關鍵點描述子 :
# 通過以上的步驟已經找到了SIFT特徵點位置、尺度和方向信息
# 下面就需要使用一組向量來描述關鍵點也就是生成特徵點描述子，這個描述符不只包含特徵點，也含有特徵點周圍對其有貢獻的像素點。
# 描述子應具有較高的獨立性，以保證匹配率。
# 特徵描述符的生成大致有三個步驟：
# 1.校正旋轉主方向，確保旋轉不變性。
# 2.生成描述子，最終形成一個128維的特徵向量
# 3.歸一化處理，將特徵向量長度進行歸一化處理，進一步去除光照的影響。
# -------------------------------------------------------------------- #
# 賦與關鍵點位置，尺度，方向確保移動，縮放，旋轉的不變性
# 還需要額外建立描述子來確保不同光線跟視角也有不變性
# 描述子會正規化成 128 維的特徵向量
# 以關鍵點周圍 16*16 的區域共 4*4 的子區域，計算 8 個方向的直方圖，共 4*4*8 = 128 維的特徵向量
# Note：每個關鍵點都會產生 128 維的特徵，而圖片會產生 N 個關鍵點，也就是會產生 (N, 128) 維度特徵
# -------------------------------------------------------------------------------------------------------------------- #
# 實作取得 SIFT 特徵-->需要轉成灰階圖片
# -------------------------------------------------------------------------------------------------------------------- #
import cv2
import numpy as np
from PIL import Image, ImageFilter
# 因為他是把點畫上去，所以我為了不讓圖片有重疊的問題，所以不斷重新定義圖片
picture = cv2.imread('lena.jpg')
picture_gray = cv2.cvtColor(picture,cv2.COLOR_BGR2GRAY)
picture_RGB1 = cv2.imread('lena.jpg')
picture_RGB = cv2.imread('lena.jpg')
picture_RGB2= cv2.imread('lena.jpg')
picture_RGB3= cv2.imread('lena.jpg')
picture_RGB4= cv2.imread('lena.jpg')
# 建立 SIFT 物件
sift1 = cv2.xfeatures2d.SIFT_create()
# 取得 SIFT 關鍵點位置
keypoints1 = sift1.detect(picture_gray, None)
img_show1 = cv2.drawKeypoints(picture_gray, keypoints1, picture)
# -------------------------------------------------------------------- #
sift2 = cv2.xfeatures2d.SIFT_create()
keypoints2 = sift2.detect(picture_RGB1, None)
img_show2 = cv2.drawKeypoints(picture_RGB1, keypoints2, picture_RGB1)
# -------------------------------------------------------------------- #
img_RGB_R = picture_RGB[...,0]
img_RGB_G = picture_RGB[...,1]
img_RGB_B = picture_RGB[...,2]
sift3 = cv2.xfeatures2d.SIFT_create()
keypoints3 = sift3.detect(img_RGB_R, None)
img_show3 = cv2.drawKeypoints(img_RGB_R, keypoints3, picture_RGB2 )
sift4 = cv2.xfeatures2d.SIFT_create()
keypoints4 = sift4.detect(img_RGB_G, None)
img_show4 = cv2.drawKeypoints(img_RGB_R, keypoints4, picture_RGB3 )
sift5 = cv2.xfeatures2d.SIFT_create()
keypoints5 = sift5.detect(img_RGB_B, None)
img_show5 = cv2.drawKeypoints(img_RGB_R, keypoints5,picture_RGB4 )
# -------------------------------------------------------------------- #
picture_all = np.hstack((picture,img_show1,img_show2,img_show3,img_show4,img_show5))
cv2.imshow('SIFT', picture_all)
cv2.waitKey(0)
cv2.destroyAllWindows()
# -------------------------------------------------------------------------------------------------------------------- #
# 每種顏色在色階的pixel 並不一樣, 所以在取feature 時, 所產生的 map 也會有差異
# 把 RGB channel 個別拆開計算 SIFT 不會得到跟灰階圖的特徵相同
# -------------------------------------------------------------------------------------------------------------------- #
img = cv2.imread('lena.jpg')
# 轉灰階圖片
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 建立 SIFT 物件
SIFT_detector = cv2.xfeatures2d.SIFT_create()

# 取得 SIFT 關鍵點位置
keypoints6 = SIFT_detector.detect(img_gray, None)
print(type(keypoints6))
#　畫圖 + 顯示圖片
img_show = cv2.drawKeypoints(img_gray, keypoints6, img)
print(type(img_show))

picture_123 = np.hstack((picture,img_show ))
while True:
    cv2.imshow('SIFT', picture_123)
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()
        break

# -------------------------------------------------------------------------------------------------------------------- #

img123456789 = cv2.imread('runway.jpg')

img_gray123456789 = cv2.cvtColor(img123456789, cv2.COLOR_BGR2GRAY)

img_sobel_x_uint8 = cv2.Sobel(img_gray123456789, -1 , dx=1, dy=0, ksize=3)

print(type(img_sobel_x_uint8))
picture_blur = cv2.GaussianBlur(img_sobel_x_uint8,(3,3),0)
picture_blur = cv2.GaussianBlur(picture_blur,(3,3),0)

SIFT_detector = cv2.xfeatures2d.SIFT_create()
keypoints123456789 = SIFT_detector.detect(picture_blur , None)
img_show123456 = cv2.drawKeypoints(picture_blur, keypoints123456789, img123456789)

SIFT_detector = cv2.xfeatures2d.SIFT_create()
keypoints12345678 = SIFT_detector.detect(picture_blur , None)
img_show12345 = cv2.drawKeypoints(img_sobel_x_uint8, keypoints12345678, img123456789)

cv2.imshow('SOBEL', picture_blur )
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('SOBEL', img_show123456 )
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('SOBEL', img_sobel_x_uint8 )
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('SOBEL', img_show12345 )
cv2.waitKey(0)
cv2.destroyAllWindows()






