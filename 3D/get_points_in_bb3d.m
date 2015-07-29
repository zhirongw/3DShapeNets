function [bbIdx, basis] = get_points_in_bb3d(points3d, bb3d)
% extract object point cloud from the 3d bounding box.
% modified from NYU code.
% points3d: point cloud from the scene.
% bb3d: object bounding box in 3d.

% each row is a orthogonal vector in basis matrix.
% column vector: x_world = basis' * x_bb + centroid;

% Order the bases.
[~, inds] = sort(abs(bb3d.basis(:,1)), 'descend');
basis = bb3d.basis(inds, :);
coeffs = bb3d.coeffs(inds);

[~, inds] = sort(abs(basis(2:3,2)), 'descend');
if inds(1) == 2
    basis(2:3,:) = flipdim(basis(2:3,:), 1);
    coeffs(2:3) = flipdim(coeffs(2:3), 2);
end

% Now, we know the basis vectors are orders X, Y, Z. Next, flip the basis
% vectors towards the viewer.
basis = flip_towards_viewer(basis, repmat(bb3d.centroid, [3 1]));

coeffs = abs(coeffs);

t_points3d = bsxfun(@minus, points3d, bb3d.centroid);
t_points3d = t_points3d * inv(basis);

bbIdx = all(bsxfun(@minus, abs(t_points3d),  coeffs)' < 0);

end

function normals = flip_towards_viewer(normals, points)
  points = points ./ repmat(sqrt(sum(points.^2, 2)), [1, 3]);
  
  proj = sum(points .* normals, 2);
  
  flip = proj > 0;
  normals(flip, :) = -normals(flip, :);
end