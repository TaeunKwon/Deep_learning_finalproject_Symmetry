# Deep_learning_finalproject_Symmetry
Dataset is on Google Drive: https://drive.google.com/file/d/1A_fLzxdGlu3_oMyq_AhCisUbIapof_Au/view?usp=sharing

batch size of autoencoder = 100

Auto Encoder Structure:
Encoder after conv1:  (100, 1350, 2)
Encoder after maxpool1:  (100, 675, 2)
Encoder after Dense1:  (100, 800)
Encoder after Dense2:  (100, 400)
Encoder after Dense3:  (100, 100)
Encoder output:  (100, 10)
Encoder after Dense1:  (100, 400)
Encoder after Dense2:  (100, 800)
Encoder after Dense3:  (100, 1300)
Decoder after upsample1:  (100, 1350, 2)
Decoder after deconv1:  (100, 2700, 1)

batch size of clustering = 20000

Trainable variables : (n_pulses) x (n_cluster), encoder parameters. 
