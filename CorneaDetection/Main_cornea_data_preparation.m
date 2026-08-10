

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Program Name: Main cornea data preperation
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Script Main cornea data preperation.m
%  Inputs:  
%        Put the patient images under input\stage\patient\image1.png
%        Ex: .\input\Extreme\ET_101_Day 21_04.16.2015\image1.png
%        Prepare data for mask R-CNN
%        
%  Outputs: Data prepared to be input for mask R-CNN in stage_test
%        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
clc;close all;clear all;

addpath '.\Scripts\';
Prepare_MaskRCNN_inference

% This function will genertate small images for cornea detection using mask
% rcnn
create_small_CNV_for_inference

