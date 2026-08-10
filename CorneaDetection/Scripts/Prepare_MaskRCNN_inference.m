%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Program Name: Prepare_MaskRCNN_inference
%                
%  Author: Yasmin Kassim
%  Copyright (C) 2020. Yasmin Kassim and K. Palaniappan and Curators of the
%                      University of Missouri, a public corporation.
%                      All Rights Reserved.
%  Created by
%  PhD student: Yasmin Kassim
%  Department of Electrical Engineering and Computer Science,
%  University of Missouri-Columbia
%  For more information, contact:
%
%      Yasmin Kassim
%      226 Engineering Building West
%      University of Missouri-Columbia
%      Columbia, MO 65211
%      ymkgz8@mail.missouri.edu
% 
% or
%      Dr. K. Palaniappan
%      329 Engineering Building West
%      University of Missouri-Columbia
%      Columbia, MO 65211
%      palaniappank@missouri.edu
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Script Prepare_MaskRCNN_inference.m
%  Inputs:  
%        Folder contains all cornea images divided in to folders (No, Mi, Mo, Ex, Se)
%%       Algorithm Details: 
%        
%  Outputs: Generate seperate folder for each image with assigning the patient name + stage as name for this folder, each foler has one image
%        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc;close all;clear all;
%% ------Files & Folders---------------------------------------------------

inputpath='..\..\Data\input\';
folders = dir(inputpath);

outputpath='..\..\Output\intermediate results\stage_test_big\';
% if (~isdir(outputpath))
%  mkdir(outputpath);
% end
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

 
   im=imread(strcat(folders2(j_j).folder,filesep,folders2(j_j).name,filesep,flist(fr).name(1:end-4),'.png'));
   image=imresize(im,0.5);

 classtype=folders(i_i).name(1:2);

   %%---- Store the image
   pathin=strcat(outputpath,filesep,folders2(j_j).name,'_',flist(fr).name(1:end-4),'_',classtype);
   if (~isdir(pathin))
      mkdir(pathin);
   end
   pathinimages=strcat(pathin,filesep,'images');
   if (~isdir(pathinimages))
      mkdir(pathinimages);
   end
   
imwrite(image,strcat(pathinimages,filesep,folders2(j_j).name,'_',flist(fr).name(1:end-4),'_',classtype,'.png'));


end

end
end
end
end
