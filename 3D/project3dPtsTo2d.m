function [points2d,z3] = project3dPtsTo2d(points3d, K, Rtilt, trans, crop)
% project 3D point cloud in world coordinate to x-y 2D plane. Provided by
% Shuran Song
% points3d: point cloud in camera coordinate
% K: camera intrinsic
% Rtilt, trans: camera extrinsic
% crop: assume it be 1

    %% inverse of get_aligned_point_cloud
    points3d = bsxfun(@plus, Rtilt * points3d', trans)';
    
    %% inverse rgb_plane2rgb_world
    if isempty(K)
        camera_params;
    else
        cx_rgb = K(1,3); cy_rgb = K(2,3);  
        fx_rgb = K(1,1); fy_rgb = K(2,2);
    end    
    % Make the original consistent with the camera location:
    x3 = points3d(:,1);
    y3 = -points3d(:,2); % when doing projection or depth->3d, always flip y dimension.
    z3 = points3d(:,3);
    
    xx = x3 * fx_rgb ./ z3 + cx_rgb;
    yy = y3 * fy_rgb ./ z3 + cy_rgb;
    
    if ~exist('crop','var')||isempty(crop)
        xx = xx - 41 + 1;
        yy = yy - 45 + 1;
    else
        xx = xx - crop(2) + 1;
        yy = yy - crop(1) + 1;
    end
    points2d = [xx yy];
end