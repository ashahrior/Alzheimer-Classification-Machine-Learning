# Alzheimer's-disease-three-stages-classification-using-Machine-Learning

### Objective
Detecting three stages of Alzheimer’s Progressive Neurodegenerative disease from 3D MRI imagery using Machine Learning. The stages are-
<ul>
  <li>AD - Alzheimer's Demented</li>
  <li>MCI - Mild Cognitive Impairment</li>
  <li>CN - Cognitively Normal</li>
</ul>

### Dataset
The used ADNI1_Annual_2_Yr_3T dataset of size 12GB (approx) was procured from <a href="adni.loni.usc.edu">ADNI</a>.

### Techs in action
<ul>
  <li>Python3</li>
  <li>Numpy</li>
  <li>OpenCV2</li>
  <li>Pandas</li>
  <li>SKLearn</li>
  <li>SKImage</li>
  <li>Anaconda</li>
</ul>

### Work process
<ul>
  <li>Converted 3D NiFTI to 3D numpy arrays</li>
  <li>Normalized the entire numpy dataset</li>
  <li>Applied CLAHE contrast amplification</li>
  <li>Performed feature extraction-</li>
  <ul>
    <li>GLCM</li>
    <li>HOG</li>
    <li>VLAD</li>
  </ul>
  <li>Performed dimensionality reduction with PCA and mRMR on the feature set</li>
  <li>Fed the ready feature-set to several machine learning algorithms such as- Random Forest, KNN, Logistic regression, LDA, SVM etc.</li>
  <li>Compared Performance analysis for each of the model</li>
</ul>

### Screenshots
<img src= "https://github.com/ashahrior/Alzheimer-Classification-Machine-Learning/blob/master/screenshots/hog_img.png" alt="HOG effect" width="500" height="150"> 
<img src="https://github.com/ashahrior/Alzheimer-Classification-Machine-Learning/blob/master/screenshots/hogimg128x64.png" alt="HOG variation" width = "350" height="150">
<img src= "https://github.com/ashahrior/Alzheimer-Classification-Machine-Learning/blob/master/screenshots/ori-vs-clahe.png" alt="CLAHE effect" width="400" height="250"> 
<img src= "https://github.com/ashahrior/Alzheimer-Classification-Machine-Learning/blob/master/screenshots/orb.png" alt="ORB descriptors" width="100" height="150"> 
