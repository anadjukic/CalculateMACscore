# CalculateMACscore

The aim of this challenge was to find a computerized method to calucalte the MAC (medial arterial calcification) using X-ray scans. This score can determine the severity of SAD (small artery disease), which in return could give prognosis of diabetic foot syndrome. In general clinical practice, this score is determined by means of angiography, however, for patient sufferic from diabetes, the administration of contrast can induce diabetic renal disease progression.

![image](https://user-images.githubusercontent.com/26152420/142758667-35cee1b5-d377-4a40-8478-0dd3d0c0b31f.png)
![image](https://user-images.githubusercontent.com/26152420/142758986-052840e0-4e40-4b7e-9d9d-014c8f3e7a12.png)

The algorithm consists of two parts:
    1. Calcified arteries detection
      1.1. Artery areas segmentation
      1.2. Five regions of interest registration
      1.3. Edge detection
      1.4. Arteries detection
    2. Caclcified arteries quantification
    
1. Fist, we wanted to be able to segment the calcified arteries that we can later quantifiy. All the processing was done on two slices of the foot: lateral (L) and anterior (A).

1.1. In this part of the algorithm, we wanted to firstly distinguish between (a) the bone and (b) the soft tissue. The soft tissue we marked as the potential region where calcified arteries can occur.

![image](https://user-images.githubusercontent.com/26152420/142759555-b1c1d3a1-87ae-4897-b8ac-f065a133c5cd.png)
![image](https://user-images.githubusercontent.com/26152420/142759560-5bec6144-bf56-4bdb-8cdf-676938a8e75a.png)

1.2. Here, we registered the two slices and the corresponding atlases (shown overlaid on the slices). The atlases were produces from a base case where the five regions of interest where labeled manually as given by instructions.
![image](https://user-images.githubusercontent.com/26152420/142759773-07598a2f-5c4e-405e-a3d9-b6e1f788b2a2.png)
![image](https://user-images.githubusercontent.com/26152420/142759786-1beddcb2-2105-4f6a-95d9-5c6ed838e2d4.png)


[1] https://www.acc.org/latest-in-cardiology/articles/2021/03/04/14/34/bad-and-sad-mac
