
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Program Name: fit_circles_to_maskrcnn_masks_results
%                
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                
%  Author: Yasmin M. Kassim & Kannappan Palaniappan
%  Copyright(C)2019-2020. Y. Kassim, K. Palaniappan and      
%			  Curators of the University of Missouri, a          
%	             public corporation.
%                 All Rights Reserved.
%
%  Created by
%  Yasmin M. Kassim & Kannappan Palaniappan
%  Department of Electrical Engineering and Computer Science,
%  University of Missouri-Columbia
%  For more information, contact:
%
%      Yasmin Kassim
%      226 Naka Hall (EBW)
%      University of Missouri-Columbia
%      Columbia, MO 65211
%      ymkgz8@mail.missouri.edu
% 
% or
%      Dr. K. Palaniappan
%      205 Naka Hall (EBW)
%      University of Missouri-Columbia
%      Columbia, MO 65211
%      palaniappank@missouri.edu
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Script fit_circles_to_maskrcnn_masks_results.m
%  Inputs:  1. Output_from_MaskRCNN_masks
%           2. Input raw images in stage_test
%        This function converts mask rcnn results masks to circular masks by fitting
% a circle using fit square error
%        
%  Outputs: Images ready to be classified in Classify_me_circles
%        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% 

clc;close all;clear all;
%% ------Files & Folders---------------------------------------------------
%
% addpath '.\Matlab_code';
addpath '.\Scripts\';
inputpath='..\..\Output\stage_test\';  %%The data here were the input to max rcnn 
inputpath_maskresults='..\..\Output\Output_from_MaskRCNN_masks\';
folders = dir(inputpath);

outputpath='..\..\Output\Classify_me_circles\';
if (~isdir(outputpath))
 mkdir(outputpath);
end

W=40*5;H=30*5;% w=200, H=150
im_masked_bigcornea=zeros(150,200,3);

%%%loop thru the folders in the path
for i_i = 1:numel(folders)

%%
if folders(i_i).isdir && ~strcmp(folders(i_i).name,'.') && ~strcmp(folders(i_i).name,'..')

folders2=dir(strcat(folders(i_i).folder,filesep, folders(i_i).name,filesep));

for j_j=1:numel(folders2)

if folders2(j_j).isdir && ~strcmp(folders2(j_j).name,'.') && ~strcmp(folders2(j_j).name,'..')
    
flist=dir(fullfile(folders2(j_j).folder,folders2(j_j).name,'*.png'));
n=length(flist);

for fr=1:n

 
   rawim=imread(strcat(folders2(j_j).folder,filesep,folders2(j_j).name,filesep,flist(fr).name(1:end-4),'.png'));
   
   if size(rawim,1)~=600 && size(rawim,1)~=800
   rawim=imresize(rawim,[600 800]);    
   end
   mask=imread(strcat(inputpath_maskresults,filesep,flist(fr).name(1:end-4),filesep,flist(fr).name(1:end-4),'_1','.png'));
   mask2 = bwmorph(mask,'remove');
   %%%%--- To print the masks before circle fitting----------- 
   
   maskbc=mask2;
   maskbc=imfill(maskbc,'holes');
   bigmaskbc=maskbc(1:H,1:W,1);
   bigmaskbc=uint8(imresize(bigmaskbc,[600,800]));
   bigmaskbc(bigmaskbc>0)=255;
   %imwrite(bigmaskbc,strcat('C:\CNV\cornea_maskrcnn_results\Matlab_code\Mask_RCNN_accuracy_CNV\Auto\before circle fitting\',folders(i_i).name,'.png'));

   %%%%-------------------------------------------------------
  [rows,cols] = find(mask2);
   XY(:,1)=rows;XY(:,2)=cols;
   Par = CircleFitByPratt(XY);
   centers(1)=Par(2);
   centers(2)=Par(1);
   radii=Par(3);

   mask_circle=zeros(600,800);
   newmask=createCirclesMask(mask_circle,centers,radii);
   
   
   bigmask=newmask(1:H,1:W,1);
   bigmask=uint8(imresize(bigmask,[600,800]));
   bigrawim(:,:,1)=rawim(1:H,1:W,1);
   bigrawim(:,:,2)=rawim(1:H,1:W,2);
   bigrawim(:,:,3)=rawim(1:H,1:W,3);

   bigrawim=uint8(imresize(bigrawim,[600,800]));
   %%---- To print the masks after circle fitting
   mybigmask=bigmask;
   mybigmask(mybigmask>0)=255;
   imwrite(mybigmask,strcat('..\..\Output\Mask_RCNN_accuracy_CNV\',folders(i_i).name,'.png'));
% %    
   if size(rawim,3)==1 
   im_masked(:,:,1)=bigrawim.*bigmask;
   im_masked(:,:,2)=bigrawim.*bigmask;
   im_masked(:,:,3)=bigrawim.*bigmask;
   else    
   im_masked(:,:,1)=bigrawim(:,:,1).*bigmask;
   im_masked(:,:,2)=bigrawim(:,:,2).*bigmask;
   im_masked(:,:,3)=bigrawim(:,:,3).*bigmask;
   end
   
   props = regionprops(bigmask, 'BoundingBox');
   thisBB = props(1).BoundingBox;

   imf=imcrop(im_masked,thisBB);
   
   imf=imresize(imf,[400 400]);
   imwrite(uint8(imf),strcat(outputpath,folders(i_i).name,'.png'));

 clear XY;
 clear bigrawim;
end

end
end
end
end
