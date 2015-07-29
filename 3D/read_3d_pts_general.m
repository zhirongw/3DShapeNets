function [rgb,points3d]=read_3d_pts_general(depthInpaint,K,depthInpaintsize,imageName,crop)
% Convert depth to point cloud. Each pixel in the depth map corresponds a
% point in 3D. Provided by Shuran Song
% depthInpaint: input depth
% K: camera intrinsic
% depthInpantsize: size(depthInpaint)
% imageName: put it emtpy(as far as I've used this function)
% crop: let it be 1

    %K is [fx 0 cx; 0 fy cy; 0 0 1];  
    %K = frames.K;
    if ~isempty(K)
        cx = K(1,3); cy = K(2,3);  
        fx = K(1,1); fy = K(2,2); 
    else
        fx = 5.19e+02;
        fy = 5.19e+02;
        cx = 320;
        cy = 240;
    end
    invalid = depthInpaint==0;
    if ~isempty(imageName)
        rgb = im2double(imageName);  
    else
        rgb = double(cat(3,zeros(depthInpaintsize(1),depthInpaintsize(2)),...
            ones(depthInpaintsize(1),depthInpaintsize(2)),...
            zeros(depthInpaintsize(1),depthInpaintsize(2))));
    end
    rgb = reshape(rgb, [], 3);
    %3D points
    [x,y] = meshgrid((1:depthInpaintsize(2))+ crop(2)-1, (1:depthInpaintsize(1))+ crop(1)-1);
    x3 = (x-cx).*depthInpaint*1/fx;  
    y3 = (y-cy).*depthInpaint*1/fy;  
    z3 = depthInpaint;       
    points3d = [x3(:) -y3(:) z3(:)];

    points3d(invalid(:),:) = NaN;
end
