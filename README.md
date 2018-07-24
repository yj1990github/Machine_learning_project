# Machine_learning_project

Files in this folder are temporarily for Hallux valgus project(CS542S)

Project7_image.py is the code for imaging processing. I have fixed the error and add desciptions. Please put it next to the h5 database

data_processing3.py is the code to generate h5 format database. After running the code, a file named as Dataset30HV.hdf5 (823.2MB) will be generate. 

Xt= np.array(f['/entry/Xt'])   #30*800*800

   30 original images
Xlt=np.array( f['/entry/Xlt'])   #120*800*800

   120 groundtruth images (1 original images produces 4 bone-label images(currently we only study four bone))
Xl=np.array(f['/entry/Xl'])    #120,

   labels(1,2,3,4,1,2,3,4,...,and so on)  
   
H= np.array(f['/entry/H'])     #30*100*100

   100*100 patches of hallux(toe) binary image
   
PP= np.array(f['/entry/PP'])   #30 200 150

   200*150 patches of proximal phalanx binary image
   
M1= np.array(f['/entry/M1'])    #(30, 400, 250)

   400*250 patches of first metatarsal bone binary image
   
M2= np.array(f['/entry/M2'])    #(30, 450, 200)

   450*200 patches of second metatarsal bone binary image
