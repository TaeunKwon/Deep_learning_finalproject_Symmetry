# Deep_learning_finalproject_Symmetry
Dataset is on Google Drive: https://drive.google.com/file/d/1A_fLzxdGlu3_oMyq_AhCisUbIapof_Au/view?usp=sharing

Auto Encoder Structure:
Encoder after conv1:  (100, 1350, 10)
Encoder after maxpool1:  (100, 675, 10)
Encoder after conv2:  (100, 225, 4)
Encoder output:  (100, 10)
Decoder after Dense:  (100, 225, 4)
Decoder after deconv1:  (100, 675, 10)
Decoder after upsample1:  (100, 1350, 10)
Decoder after deconv2:  (100, 2700, 4)