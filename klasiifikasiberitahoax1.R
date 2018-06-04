library(caret)
library(tm)
library(SnowballC)
library(arm)
# Training data.
data <- c('Bom mapolres surabaya adalah pengalihan isu.', 'Peledakan bom mapolres surabaya menelan 5 korban jiwa.', 'Sebarkan Tulisan soal Bom Surabaya, Dosen USU Menyesal','Misa Penghormatan Bayu Wardhana, Korban Bom Bunuh Diri Gereja Surabaya','3 Alasan Istri dan Anak Terlibat Terorisme Jadi Pengantin Bom Bunuh Diri','3 Orang Ini Ditangkap karena Sebut Bom Surabaya sebagai Pengalihan Isu','Mengenang Giri, Sosok Pencegat Mobil Berisi Bom Gereja Surabaya','Menko Polhukam: Ibadah Minggu di Seluruh Gereja di Indonesia Aman','Pastor Gereja Santa Maria Tak Bercela Minta Jemaat Ampuni Pelaku Bom','Isak Tangis Keluarga Iringi Pemakaman Korban Bom Gereja Surabaya','Alasan Tersangka Peracik Bom Unri Tak Penuhi Permintaan Penyerang Mapolda Riau','Polisi: Penyerang Mapolda Pesan Bom ke Terduga Teroris di Universitas Riau','Kasus Temuan Bom Rakitan, Universitas Riau Berterima Kasih pada Densus 88','Menkumham Beri Semangat Jemaat Gereja Sasaran Bom Surabaya','Alasan Sepele yang Picu Hoaks Teror Bom di Kereta Merak-Rangkasbitung','Teror Bom Kereta Rangkasbitung-Merak Ternyata Hoaks','Menhub Sepakat Blacklist Penumpang yang Bercanda Bom','Tiga Makam untuk 7 Jenazah Terduga Teroris Bom di Surabaya' ,'Bom di Universitas Riau Akan Diledakkan di DPRD dan DPR','Terduga Teroris di Univeritas Riau Belajar Merakit Bom dari Instagram')
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
data2 <- c( 'Jatuh korban jiwa dalam ledakan bom di mapolres surabaya', 'Peledakan Bom Surabaya terjadi di Gereja,Korban Bom Bernama Bayu Wardhana' ,'Tiga Makam untuk 7 Jenazah Terduga Teroris Bom di Surabaya','Polisi: Pesan Berantai Sebut Jakarta Rawan Bom Adalah Hoax','Kereta Jurusan Rangkasbitung-Merak Diteror Bom','Bom Rakitan di Universitas Riau Punya Daya Ledak Sama dengan Teror di Surabaya','Insiden Lelucon Bom di Pesawat Lion Air Disorot Dunia','Seperti Suara Bom, Ledakan Petasan di Lawang Malang Tewaskan 1 Warga','Duar, Ledakan di Sidang Pleidoi Aman Abdurrahman Bukan Bom','JAWA TIMUR13 hari laluCara Risma Pulihkan Trauma Anak Usai Bom Surabaya')
corpus <- VCorpus(VectorSource(data2))
tdm <- DocumentTermMatrix(corpus, control = list(dictionary = Terms(tdm), removePunctuation = TRUE, stopwords = TRUE, stemming = TRUE, removeNumbers = TRUE))
test <- as.matrix(tdm)

# Check accuracy on test.
predict(fit, newdata = test)
