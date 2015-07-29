% Projects the depth points from the image plane to the 3D world
% coordinates.
%
% Args:
%   imgDepth - depth map which has already been projected onto the RGB
%              image plane, an HxW matrix where H and W are the height and
%              width of the matrix, respectively.
%
% Returns:
%   points3d - the point cloud in the world coordinate frame, an Nx3
%              matrix.
%
% Author: Nathan Silberman (silberman@cs.nyu.edu)
function points3d = rgb_plane2rgb_world(imgDepth)
  camera_params;
  
  imgDepthOrig = imgDepth;
  % Make the original consistent with the camera location:
  [xx, yy] = meshgrid(1:640, 1:480);

  x3 = (xx - cx_rgb) .* imgDepthOrig / fx_rgb;
  y3 = (yy - cy_rgb) .* imgDepthOrig / fy_rgb;
  z3 = imgDepthOrig;
  
  points3d = [x3(:) -y3(:) z3(:)];
end