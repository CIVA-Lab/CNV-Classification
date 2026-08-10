To get cornea detection

1. Put your raw images in a folder called input, the images should be placed as this example:

``` input\Extreme\ET_101_Day 21_04.16.2015\image1.png ```

2. Run ```Main_cornea_data_preparation.m```

This script will prepare your input for Mask R-CNN detection.

3. Setup [Mask R-CNN](https://github.com/matterport/Mask_RCNN):  
Put the ```mosaic_cornea_weights``` in log, and put your ```Output\stage_test``` folder in the same folder with the nucleus example and run ```nucleus.py```

4. Take the detection masks and place them in 

```Output/Output_from_MaskRCNN_masks```

To treat the results generated from mask R-CNN by fitting a circle on mask R-CNN results, you should run 5, and the results will be ready in ```Output/Classify_me_circles``` folder for RF CNV Grading

5. ``` fit_circles_to_maskrcnn_masks_results.m ```
 
This script uses [Pratt method](https://www.mathworks.com/matlabcentral/fileexchange/22643-circle-fit-pratt-method) to fit the cicle

### Project Collaborators and Contact

**Author:** Yasmin M. Kassim and Kannappan Palaniappan

Copyright &copy; 2019-2020. Yasmin Kassim and Prof. K. Palaniappan and Curators of the University of Missouri, a public corporation. All Rights Reserved.

**Created by:** Ph.D. student: Yasmin Kassim  
Department of Electrical Engineering and Computer Science,  
University of Missouri-Columbia  

For more information, contact:

* **Yasmin Kassim**  
226 Naka Hall (EBW)  
University of Missouri-Columbia  
Columbia, MO 65211  
ymkgz8@mail.missouri.edu  


* **Dr. K. Palaniappan**  
205 Naka Hall (EBW)  
University of Missouri-Columbia  
Columbia, MO 65211  
palaniappank@missouri.edu  
