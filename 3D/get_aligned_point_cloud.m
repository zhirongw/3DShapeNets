% Aligns the point cloud given a rotation matrix.
%
% Args:
%   points3d - Nx3 point cloud.
%   R - 3x3 rotation matrix.
%
% Returns:
%   points3d - Nx3 aligned point cloud.
%
% Author: Nathan Silberman (silberman@cs.nyu.edu)
function points3d = get_aligned_point_cloud(points3d, R)
  points3d = (R * points3d')';
end