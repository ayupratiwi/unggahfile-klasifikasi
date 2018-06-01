library(caret)
library(tm)
library(SnowballC)
library(arm)
# Training data.
data <- c('Bom mapolres surabaya adalah pengalihan isu.', 'Peledakan bom mapolres surabaya menelan 5 korban jiwa.', 'Sebarkan Tulisan soal Bom Surabaya, Dosen USU Menyesal','Misa Penghormatan Bayu Wardhana, Korban Bom Bunuh Diri Gereja Surabaya','3 Alasan Istri dan Anak Terlibat Terorisme Jadi Pengantin Bom Bunuh Diri','3 Orang Ini Ditangkap karena Sebut Bom Surabaya sebagai Pengalihan Isu','Mengenang Giri, Sosok Pencegat Mobil Berisi Bom Gereja Surabaya','Menko Polhukam: Ibadah Minggu di Seluruh Gereja di Indonesia Aman','Pastor Gereja Santa Maria Tak Bercela Minta Jemaat Ampuni Pelaku Bom','Isak Tangis Keluarga Iringi Pemakaman Korban Bom Gereja Surabaya','Jurus Waskita Atasi Macet di Tol Pejagan-Pemalang','Xiaomi Resmi Luncurkan Mi 8 SE, Smartphone Perdana dengan Snapdragon 710','Kemenhub: Ada Diskon Tarif Bakal Bikin Pemudik Pakai Jalan Tol','Hutan yang Terbakar Saat Letusan Gunung Merapi Sudah Padam','John Paul Ivan Tidak Akan Berdamai dengan Pembuat Hoax','Ini Identitas Kakek di KRL yang Mirip Mendiang Soeharto','Pidato Prabowo di Acara Buruh Sempat Terhenti, Kenapa?','Pemain City Bela Tato Kontroversial Sterling','Jangan Salahkan Sistem Zonasi Sekolah karena Kasus Bunuh Diri Pelajar Blitar','Curangi Konsumen, Pengelola SPBU di Nagreg Kurangi Takaran BBM')
corpus <- VCorpus(VectorSource(data))

# Create a document term matrix.
tdm <- DocumentTermMatrix(corpus, list(removePunctuation = TRUE, stopwords = TRUE, stemming = TRUE, removeNumbers = TRUE))

# Convert to a data.frame for training and assign a classification (factor) to each document.
train <- as.matrix(tdm)
train <- cbind(train, c(0, 1))
colnames(train)[ncol(train)] <- 'y'
train <- as.data.frame(train)
train$y <- as.factor(train$y)
data
train 
# Train.
fit <- train(y ~ ., data = train, method = 'bayesglm')

# Check accuracy on training.
predict(fit, newdata = train)

# Test data.
data2 <- c( 'Jatuh korban jiwa dalam ledakan bom di mapolres surabaya', 'Peledakan Bom Surabaya terjadi di Gereja,Korban Bom Bernama Bayu Wardhana' ,'Tiga Makam untuk 7 Jenazah Terduga Teroris Bom di Surabaya' ,'Viral Nomor IMEI Ponsel Bisa Disadap, Kemkominfo: Itu Hoax' ,'Polri Ancam  Tindak Penyebar Video Hoax Letusan Gunung Merapi' ,'Menteri PANRB: Usut Tuntas Hoax Penerimaan CPNS 2018','Xiaomi Mi 8 Resmi Meluncur dalam 3 Versi,ini Spesifikasi dan Harganya','Letusan Hari Ini Bukan Awan Panas, Status Gunung Merapi Masih Waspada','Polisi: Kabar Kader HMI MPO Meninggal Saat Unjuk Rasa di Monas, Hoax','Cara Jitu Setop Kebiasaan Buruk')
corpus <- VCorpus(VectorSource(data2))
tdm <- DocumentTermMatrix(corpus, control = list(dictionary = Terms(tdm), removePunctuation = TRUE, stopwords = TRUE, stemming = TRUE, removeNumbers = TRUE))
test <- as.matrix(tdm)

# Check accuracy on test.
predict(fit, newdata = test)
